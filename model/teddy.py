from typing import Any
import torch
import string

import pickle
from torch import nn
import numpy as np
from torchvision import models
from torchvision.utils import save_image, make_grid

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .cnn_decoder import FCNDecoder
from .ocr import OrigamiNet
from .hwt.model import Discriminator as HWTDiscriminator
from util.functional import Clock, MetricCollector
from .content_modules import UnifontModule, ConvUnifontModule, OnehotModule, UnifontShiftTransform


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


class CTCLabelConverter(nn.Module):
    def __init__(self, charset):
        super(CTCLabelConverter, self).__init__()
        self.charset = sorted(set(charset))

        # NOTE: 0 is reserved for 'blank' token required by CTCLoss
        self.dict = {char: i + 1 for i, char in enumerate(self.charset)}
        self.charset.insert(0, '[blank]')  # dummy '[blank]' token for CTCLoss (index 0)

    def encode(self, labels, device='cpu'):
        assert set(''.join(labels)) <= set(self.charset), f'The following character are not in charset {set("".join(labels)) - set(self.charset)}'
        length = torch.LongTensor([len(lbl) for lbl in labels])
        labels = [torch.LongTensor([self.dict[char] for char in lbl]) for lbl in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
        return labels.to(device), length.to(device)

    def decode(self, labels, length):
        texts = []
        assert len(labels) == len(length)
        for lbl, lbl_len in zip(labels, length):
            char_list = []
            for i in range(lbl_len):
                if lbl[i] != 0 and (not (i > 0 and lbl[i - 1] == lbl[i])) and lbl[i] < len(self.charset):  # removing repeated characters and blank.
                    char_list.append(self.charset[lbl[i]])
            texts.append(''.join(char_list))
        return texts

    def decode_batch(self, preds):
        preds = rearrange(preds, 'w b c -> b w c')
        _, preds_index = preds.max(2)
        preds_index = preds_index.cpu().numpy()
        preds_size = preds.size(1) - (np.flip(preds_index, 1) > 0).argmax(-1)
        preds_size = np.where(preds_size < preds.size(1), preds_size, 0)
        return self.decode(preds_index, preds_size)


class NoiseExpansion(nn.Module):
    def __init__(self, expansion_factor=1, noise_alpha=0.1):
        super().__init__()
        self.expansion_factor = expansion_factor
        self.noise_alpha = noise_alpha

    def forward(self, x):
        x = repeat(x, 'b l d -> (b e) l d', e=self.expansion_factor)
        noise = torch.randn_like(x) * self.noise_alpha
        x += noise
        return x


class TeddyGenerator(nn.Module):
    def __init__(self, image_size, patch_size, dim=512, depth=3, heads=8, mlp_dim=512,
                 expansion_factor=1, noise_alpha=0.0, channels=3, num_style=3, dropout=0.1,
                 emb_dropout=0.1, embedding_module='UnifontModule', embedding_module_kwargs={}):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.patch_width = patch_width

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h (p pw) -> b p (h pw c)', pw=self.patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + num_style, dim))
        self.style_tokens = nn.Parameter(torch.randn(1, num_style, dim))

        assert embedding_module in globals(), f'Embedding module {embedding_module} not found.'
        embedding_module = globals()[embedding_module]
        embedding_module_kwargs['dim'] = dim
        self.style_embedding = embedding_module(**embedding_module_kwargs)
        self.gen_embedding = embedding_module(**embedding_module_kwargs)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)

        encoder_norm = nn.LayerNorm(dim)
        decoder_norm = nn.LayerNorm(dim)

        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=depth, norm=encoder_norm)
        self.transformer_style_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=depth, norm=decoder_norm)
        self.transformer_gen_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=depth, norm=decoder_norm)

        self.batch_expansion = NoiseExpansion(expansion_factor, noise_alpha)
        self.cnn_decoder = FCNDecoder(dim=dim, out_dim=channels)
        self.rearrange_expansion = Rearrange('(b e) c h w -> b e c h w', e=expansion_factor)

    def forward_style(self, style_imgs, style_tgt):
        x = self.to_patch_embedding(style_imgs)
        b, n, _ = x.shape

        assert x.shape[1] <= self.pos_embedding.shape[1], f'Number of style tokens {x.shape[1]} > positional embeddings {self.pos_embedding.shape[1]}. {style_imgs.shape=}'
        x += self.pos_embedding[:, :n]

        x = self.transformer_encoder(x)

        style_tgt = self.style_embedding(style_tgt)
        style_tokens = self.style_tokens.repeat(b, 1, 1)
        style_tgt = torch.cat((style_tokens, style_tgt), dim=1)
        x = self.transformer_style_decoder(style_tgt, x)
        return x

    def forward_gen(self, style_emb, gen_tgt):
        gen_tgt = self.gen_embedding(gen_tgt)
        x = self.transformer_gen_decoder(gen_tgt, style_emb)

        x = self.batch_expansion(x)
        x = self.cnn_decoder(x)
        x = self.rearrange_expansion(x)
        return x

    def forward(self, style_imgs, style_tgt, gen_tgt):
        style_emb = self.forward_style(style_imgs, style_tgt)
        fakes = self.forward_gen(style_emb, gen_tgt)
        return fakes


class ResnetDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(num_classes=1)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, src):
        return self.resnet(src)


class FontSquareEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg16(num_classes=10400)
        checkpoint = torch.hub.load_state_dict_from_url('https://github.com/aimagelab/font_square/releases/download/VGG-16/VGG16_class_10400.pth')
        self.model.load_state_dict(checkpoint)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.classifier = nn.Identity()

    def forward(self, src):
        return self.model(src)


class ImageNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.classifier = nn.Identity()

    def forward(self, src):
        return self.model(src)


class RandPatchSampler:
    def __init__(self, patch_width=16, patch_num=4, img_channels=1):
        self.patch_width = patch_width
        self.patch_num = patch_num
        self.img_to_seq = Rearrange('b c h (p pw) -> b p (h pw c)', pw=self.patch_width)
        self.seq_to_img = Rearrange('b p (h pw c) -> b c h (p pw)', pw=self.patch_width, c=img_channels)

    def __call__(self, img, img_len=None):
        b, c, h, w = img.shape
        device = img.device
        if img_len is None:
            img_len = torch.tensor([w] * b, device=device)
        img_len = img_len // self.patch_width
        img_seq = self.img_to_seq(img[:, :, :, :img_len.max() * self.patch_width])
        rand_idx = torch.randint(img_len.max() - 1, (img_len.size(0), self.patch_num)).to(device)
        rand_idx %= img_len.unsqueeze(-1)
        rand_idx += torch.arange(rand_idx.size(0), device=device).unsqueeze(-1) * img_seq.size(1)
        rand_idx = rand_idx.flatten()
        batch2flat = Rearrange('b l d -> (b l) d')
        flat2batch = Rearrange('(b l) d -> b l d', b=b)
        img_seq = flat2batch(batch2flat(img_seq)[rand_idx])
        # return self.seq_to_img(img_seq)
        return img_seq.reshape((-1, 1, 32, self.patch_width))


class PatchSampler:
    def __init__(self, patch_width, patch_count, unit=16):
        assert patch_width % unit == 0, f'Patch width must be divisible by {unit}'
        self.patch_width = patch_width
        self.patch_count = patch_count
        self.unit = unit
        self.img_to_seq = Rearrange('b c h (p u) -> b p (h u c)', u=unit)

    def __call__(self, img, img_len=None):
        b, c, h, w = img.shape
        if w < self.patch_width:
            pad_width = self.patch_width - w
            img = torch.nn.functional.pad(img, (0, pad_width), value=1)
            b, c, h, w = img.shape

        device = img.device
        img_len = torch.tensor([w] * b, device=device) if img_len is None else img_len
        img_len = (img_len / self.unit).ceil().long()
        img_seq = self.img_to_seq(img[:, :, :, :img_len.max() * self.unit])
        rand_idx = torch.randint(img_len.max() + 1 - (self.patch_width // self.unit), (b, self.patch_count)).to(device)
        rand_idx %= img_len.unsqueeze(-1)
        rand_idx += torch.arange(b, device=device).unsqueeze(-1) * img_seq.size(1)
        rand_idx = rand_idx.flatten()
        batch2flat = Rearrange('b l d -> (b l) d')
        flat2batch = Rearrange('(b l) d -> b l d', b=b)
        imgs = []
        flat_img_seq = batch2flat(img_seq)
        for i in range(self.patch_width // self.unit):
            tmp = flat2batch(flat_img_seq[rand_idx + i])
            tmp = rearrange(tmp, 'b p (h u c) -> b p c h u', h=h, u=self.unit)
            imgs.append(tmp)
        # return self.seq_to_img(img_seq)
        return rearrange(torch.cat(imgs, -1), 'b p c h pw -> (b p) c h pw')


class TeddyDiscriminator(torch.nn.Module):
    def __init__(self, patch_width, charset):
        super().__init__()
        self.dis_local = HWTDiscriminator(resolution=patch_width, vocab_size=len(charset) + 1)
        # self.dis_local = ResnetDiscriminator()
        self.dis_global = HWTDiscriminator(resolution=patch_width, vocab_size=len(charset) + 1)


class Teddy(torch.nn.Module):
    def __init__(self, charset, img_height, img_channels, gen_dim, dis_dim, gen_max_width, gen_patch_width,
                 gen_expansion_factor, gen_emb_module, gen_emb_shift, dis_patch_width, dis_patch_count, style_patch_width,
                 style_patch_count, **kwargs) -> None:
        super().__init__()
        self.expansion_factor = gen_expansion_factor
        self.text_converter = CTCLabelConverter(charset)
        self.ocr = OrigamiNet(o_classes=len(charset) + 1)
        self.style_encoder = FontSquareEncoder()
        self.apperence_encoder = ImageNetEncoder()

        freeze(self.style_encoder)
        embedding_module_kwargs = {'charset': charset, 'transforms': UnifontShiftTransform(gen_emb_shift)}
        self.generator = TeddyGenerator((img_height, gen_max_width), (img_height, gen_patch_width), dim=gen_dim, expansion_factor=gen_expansion_factor,
                                        channels=img_channels, embedding_module=gen_emb_module, embedding_module_kwargs=embedding_module_kwargs)
        # self.discriminator = ResnetDiscriminator()
        # self.discriminator = TeddyDiscriminator((img_height, dis_patch_width * dis_patch_count), (img_height, gen_patch_width), dim=dis_dim,
        #                                         expansion_factor=gen_expansion_factor, channels=img_channels)
        # self.discriminator = TeddyDiscriminator(gen_patch_width, charset)
        self.discriminator = HWTDiscriminator(resolution=gen_patch_width, vocab_size=len(charset) + 1)

        # self.dis_patch_sampler = RandPatchSampler(patch_width=dis_patch_width, patch_num=dis_patch_count, img_channels=img_channels)
        # self.style_patch_sampler = RandPatchSampler(patch_width=style_patch_width, patch_num=style_patch_count, img_channels=img_channels)
        self.dis_patch_sampler = PatchSampler(dis_patch_width, dis_patch_count)
        self.style_patch_sampler = PatchSampler(style_patch_width, style_patch_count)
        self.collector = MetricCollector()

    def generate(self, gen_texts, style_texts, style_imgs, enc_style_text=None, enc_gen_text=None):
        device = style_imgs.device

        if enc_style_text is None or enc_gen_text is None:
            enc_style_text, _ = self.text_converter.encode(style_texts, device)
            enc_gen_text, _ = self.text_converter.encode(gen_texts, device)

        src_style_emb = self.generator.forward_style(style_imgs, enc_style_text)
        fakes = self.generator.forward_gen(src_style_emb, enc_gen_text)
        return fakes

    def forward(self, batch):
        device = next(self.parameters()).device
        enc_style_text, enc_style_text_len = self.text_converter.encode(batch['style_texts'], device)
        enc_gen_text, enc_gen_text_len = self.text_converter.encode(batch['gen_texts'], device)
        fakes = self.generate(batch['gen_texts'], batch['style_texts'], batch['style_imgs'], enc_style_text, enc_gen_text)

        dis_glob_real_pred = self.discriminator(batch['style_imgs'])
        dis_glob_fake_pred = self.discriminator(fakes[:, 0])

        real_samples = self.dis_patch_sampler(batch['style_imgs'], img_len=batch['style_imgs_len'])
        fake_samples = self.dis_patch_sampler(fakes[:, 0], img_len=enc_gen_text_len * 16)
        dis_local_real_pred = self.discriminator(real_samples)
        dis_local_fake_pred = self.discriminator(fake_samples)

        fakes_whole = repeat(fakes, 'b e 1 h w -> (b e) 3 h w')
        real_whole = repeat(batch['style_imgs'], 'b 1 h w -> b 3 h w')
        other_whole = repeat(batch['other_author_imgs'], 'b 1 h w -> b 3 h w')
        enc_gen_text = repeat(enc_gen_text, 'b w -> (b e) w', e=self.expansion_factor)
        enc_gen_text_len = repeat(enc_gen_text_len, 'b -> (b e)', e=self.expansion_factor)

        style_glob_fakes = self.style_encoder(fakes_whole)
        style_glob_negative = self.style_encoder(other_whole)
        style_glob_positive = self.style_encoder(real_whole)

        real_samples = self.style_patch_sampler(real_whole, img_len=batch['style_imgs_len'])
        fake_samples = self.style_patch_sampler(fakes_whole, img_len=enc_gen_text_len * 16)
        other_samples = self.style_patch_sampler(other_whole, img_len=batch['other_author_imgs_len'])

        style_local_real = self.style_encoder(real_samples)
        style_local_fakes = self.style_encoder(fake_samples)
        style_local_other = self.style_encoder(other_samples)

        appea_local_real = self.apperence_encoder(real_samples)
        appea_local_fakes = self.apperence_encoder(fake_samples)
        appea_local_other = self.apperence_encoder(other_samples)

        ocr_fake_pred = self.ocr(fakes_whole)

        ocr_real_pred = None
        if 'ocr_real_train' in batch and batch['ocr_real_train']:
            ocr_real_pred = self.ocr(real_whole)
        elif 'ocr_real_eval' in batch and batch['ocr_real_eval']:
            with torch.inference_mode():
                ocr_real_pred = self.ocr(real_whole)

        results = {
            'fakes': fakes,
            'dis_glob_real_pred': dis_glob_real_pred,
            'dis_glob_fake_pred': dis_glob_fake_pred,
            'dis_local_real_pred': dis_local_real_pred,
            'dis_local_fake_pred': dis_local_fake_pred,
            # 'same_other_pred': same_other_pred,
            'ocr_fake_pred': ocr_fake_pred,
            'ocr_real_pred': ocr_real_pred,
            # 'src_style_emb': src_style_emb,
            # 'gen_style_emb': gen_style_emb,
            'enc_gen_text': enc_gen_text,
            'enc_gen_text_len': enc_gen_text_len,
            'enc_style_text': enc_style_text,
            'enc_style_text_len': enc_style_text_len,
            'style_local_fakes': style_local_fakes,
            'style_local_real': style_local_real,
            'style_local_other': style_local_other,
            'appea_local_fakes': appea_local_fakes,
            'appea_local_real': appea_local_real,
            'appea_local_other': appea_local_other,
            'real_samples': real_samples,
            'fake_samples': fake_samples,
            'style_glob_fakes': style_glob_fakes,
            'style_glob_negative': style_glob_negative,
            'style_glob_positive': style_glob_positive,
        }
        return results

    def generate_eval_page(self, gen_texts, style_texts, style_imgs, max_batch_size=4):
        gen_texts = gen_texts[:max_batch_size]
        style_texts = style_texts[:max_batch_size]
        style_imgs = style_imgs[:max_batch_size]
        batch_size = len(gen_texts)

        alphabet = list(string.ascii_letters + string.digits)

        fakes = []
        fakes.append(self.generate(style_texts, style_texts, style_imgs))  # fakes_same_text
        fakes.append(self.generate([''.join(alphabet)] * batch_size, style_texts, style_imgs))  # fakes_alph_join
        alph_sep = ' '.join(alphabet)
        fakes.append(self.generate([alph_sep[:len(alph_sep) // 2 + 1]] * batch_size, style_texts, style_imgs))  # fakes_alph_sep_first_half
        fakes.append(self.generate([alph_sep[len(alph_sep) // 2:]] * batch_size, style_texts, style_imgs))  # fakes_alph_sep_second_half
        fakes.append(self.generate(gen_texts, style_texts, style_imgs))  # fakes_1
        # fakes.append(self.generate(gen_texts, style_texts, style_imgs))  # fakes_2
        # fakes.append(self.generate(gen_texts, style_texts, style_imgs))  # fakes_3
        # fakes.append(self.generate(gen_texts, style_texts, style_imgs))  # fakes_4
        fakes.append(self.generate([txt.split()[0] for txt in gen_texts], style_texts, style_imgs))  # fakes_short
        fakes.append(self.generate([txt + ' ' + txt for txt in gen_texts], style_texts, style_imgs))  # fakes_long

        max_width = max([fake.shape[-1] for fake in fakes])
        fakes = [torch.nn.functional.pad(fake, (0, max_width - fake.shape[-1]), value=1) for fake in fakes]
        fakes = torch.cat(fakes, dim=-2)[:, 0]

        max_width = max([img.shape[-1] for img in style_imgs])
        max_height = max([fake.shape[-2] for fake in fakes])
        style_imgs = [torch.nn.functional.pad(img, (0, max_width - img.shape[-1], 0, max_height - img.shape[-2]), value=1) for img in style_imgs]
        style_imgs = torch.stack(style_imgs)

        return make_grid(torch.cat((style_imgs, fakes), dim=-1), nrow=1)
