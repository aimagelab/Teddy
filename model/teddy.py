from typing import Any
import torch
import string
import math
import warnings

import pickle
from torch import nn
import numpy as np
from torchvision import models
from torchvision.utils import save_image, make_grid
from torchvision.transforms import functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .vae import Decoder, Sampler
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


class PaddingCat(nn.Module):
    def __init__(self, padding_value=1, cat_dim=-1):
        super().__init__()
        self.padding_value = padding_value
        self.cat_dim = cat_dim

    def forward(self, real, fake):
        real_width = real.size(-1)
        fake_width = fake.size(-1)
        pad = (0, 0, abs(real_width - fake_width), 0)

        if real_width > fake_width:
            fake = F.pad(fake, pad, fill=self.padding_value)
        elif fake_width > real_width:
            real = F.pad(real, pad, fill=self.padding_value)

        return torch.cat((real, fake), dim=self.cat_dim)


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
    

class ConvTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear1 = self._linear_to_conv(kernel_size, self.linear1)
        self.linear2 = self._linear_to_conv(kernel_size, self.linear2)

    @staticmethod
    def _linear_to_conv(kernel_size, linear):
        return nn.Sequential(
            Rearrange('b l c -> b c l'),
            nn.Conv1d(linear.in_features, linear.out_features, kernel_size, padding=kernel_size // 2),
            Rearrange('b c l -> b l c')
        )
    

class NoCrossTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def _mha_block(self, x, *args, **kwargs):
        return x      


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        assert x.size(1) <= self.positional_embedding.size(1), f'Input sequence length {x.size(1)} is greater than the maximum positional embedding length {self.positional_embedding.size(1)}'
        x = x + self.positional_embedding[:, :x.size(1)]
        return x


class TeddyDiscriminator(nn.Module):
    def __init__(self, image_size, patch_size, dim=512, depth=6, heads=8, mlp_dim=2048, channels=1, dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.patch_width = patch_width

        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h (p pw) -> b p (h pw c)', pw=self.patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_encoding = LearnedPositionalEncoding(dim, max_len=image_width // patch_width * 2)
        self.out_token = nn.Parameter(torch.randn(1, 1, dim))

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        encoder_norm = nn.LayerNorm(dim)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=depth, norm=encoder_norm)

        self.linear = nn.Linear(dim, 1)

    def forward(self, src):
        x = self.to_patch_embedding(src)
        x = self.pos_encoding(x)
        b, n, _ = x.shape

        out_token = self.out_token.repeat(b, 1, 1)
        x = torch.cat((out_token, x), dim=1)

        x = self.transformer_encoder(x)
        x = x[:, 0, :]
        return self.linear(x)


class TeddyGenerator(nn.Module):
    def __init__(self, image_size, patch_size, dim=512, depth=6, heads=8, mlp_dim=2048, channels=3, num_style=3,
                 dropout=0.1, embedding_module='UnifontModule', embedding_module_kwargs={}, cnn_decoder_width=16):
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

        self.img_pos_encoding = LearnedPositionalEncoding(dim)
        self.style_pos_encoding = LearnedPositionalEncoding(dim)
        self.gen_pos_encoding = lambda x: x

        self.style_tokens = nn.Parameter(torch.randn(1, num_style, dim))

        assert embedding_module in globals(), f'Embedding module {embedding_module} not found.'
        embedding_module = globals()[embedding_module]
        embedding_module_kwargs['dim'] = dim
        self.style_embedding = embedding_module(**embedding_module_kwargs)
        self.gen_embedding = embedding_module(**embedding_module_kwargs)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        # transformer_decoder_layer = ConvTransformerDecoderLayer(3, d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        # transformer_decoder_layer = NoCrossTransformerDecoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)

        encoder_norm = nn.LayerNorm(dim)
        decoder_norm = nn.LayerNorm(dim)

        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=depth, norm=encoder_norm)
        self.transformer_style_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=depth, norm=decoder_norm)
        self.transformer_gen_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=depth, norm=decoder_norm)

        self.cnn_decoder = Decoder(in_dim=dim, out_dim=channels, width=cnn_decoder_width // 8)
        # self.vae.requires_grad_(False)
        # self.decoder = FCNDecoder(dim=dim, out_dim=channels)

    def forward_style(self, style_img, style_tgt):
        x = self.to_patch_embedding(style_img)
        b, n, _ = x.shape

        x = self.img_pos_encoding(x)
        memory = self.transformer_encoder(x)

        style_tgt = self.style_embedding(style_tgt)
        style_tgt = self.style_pos_encoding(style_tgt)
        style_tgt_len = style_tgt.size(1)

        style_tokens = self.style_tokens.repeat(b, 1, 1)
        style_tokens_len = style_tokens.size(1)
        style_tgt = torch.cat((style_tokens, style_tgt), dim=1)
        style_memory = self.transformer_style_decoder(style_tgt, memory)

        return style_memory.split((style_tokens_len, style_tgt_len), dim=1)

    def forward_gen(self, style_memory, gen_tgt):
        gen_tgt = self.gen_embedding(gen_tgt)
        gen_tgt = self.gen_pos_encoding(gen_tgt)

        output = self.transformer_gen_decoder(gen_tgt, style_memory)
        output = self.cnn_decoder(output)
        return output

    def forward(self, style_img, style_tgt, gen_tgt):
        style_memory, _ = self.forward_style(style_img, style_tgt)
        fakes = self.forward_gen(style_memory, gen_tgt)
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
            img_len = torch.clamp_min(img_len, self.patch_width)
            img = torch.nn.functional.pad(img, (0, self.patch_width - w), value=1)
            b, c, h, w = img.shape

        device = img.device
        img_len = torch.tensor([w] * b, device=device) if img_len is None else img_len
        img_seq = self.img_to_seq(img[:, :, :, :img_len.max() * self.unit])

        try:
            rand_idx = torch.randint(img_len.max() + 1 - (self.patch_width // self.unit), (b, self.patch_count)).to(device)
        except RuntimeError:
            warnings.warn(f'img_len={img_len.tolist()}, {img_len.max()=} img_seq={img_seq.shape} {self.patch_width//self.unit=}')
            rand_idx = torch.zeros((b, self.patch_count)).long().to(device)
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
    

class UnfoldPatchSampler:
    def __init__(self, patch_width, patch_height, patch_count, stride=16):
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.patch_count = patch_count
        self.stride = stride
        self.unfold = nn.Unfold(kernel_size=(patch_height, patch_width), stride=stride)
    
    def __call__(self, img, img_len):
        b, c, h, w = img.shape
        if w < self.patch_width:
            img = torch.nn.functional.pad(img, (0, self.patch_width - w), value=1)
            b, c, h, w = img.shape

        max_idx = ((img_len - self.patch_width) / self.stride).clamp_min(0)
        patches = rearrange(self.unfold(img), 'b (c h w) l -> b l c h w', c=c, h=h)
        rand_idx = torch.randint(patches.size(1), (b, self.patch_count)).to(img.device)
        rand_idx %= max_idx.long().unsqueeze(1)
        rand_idx = torch.stack([rand_idx, repeat(max_idx, 'b -> b p', p=self.patch_count)]).min(0).values
        rand_idx += torch.arange(b, device=img.device).unsqueeze(1) * patches.size(1)

        return patches.flatten(0, 1)[rand_idx.flatten().long()]
[]    

class StartPatchSampler(UnfoldPatchSampler):    
    def __call__(self, img, img_len):
        b, c, h, w = img.shape
        if w < self.patch_width:
            img = torch.nn.functional.pad(img, (0, self.patch_width - w), value=1)
            b, c, h, w = img.shape

        patches = img[..., :self.patch_width]
        return repeat(patches, 'b c h w -> (b p) c h w', p=self.patch_count)


class Teddy(torch.nn.Module):
    def __init__(self, charset, img_height, img_channels, gen_dim, dis_dim, gen_max_width, gen_patch_width, gen_expansion_factor,
                 gen_emb_module, gen_emb_shift, gen_glob_style_tokens, dis_patch_width, dis_patch_count, style_patch_width,
                 style_patch_count, gen_cnn_decoder_width, wid_num_authors, single_img_dis, **kwargs) -> None:
        super().__init__()
        self.expansion_factor = gen_expansion_factor
        self.gen_cnn_decoder_width = gen_cnn_decoder_width

        self.text_converter = CTCLabelConverter(charset)
        self.ocr = OrigamiNet(o_classes=len(charset) + 1) if kwargs['weight_ocr'] > 0 else None
        self.style_encoder = FontSquareEncoder() if kwargs['weight_style_local'] > 0 or kwargs['weight_style_global'] > 0 else None
        self.apperence_encoder = ImageNetEncoder() if kwargs['weight_appea_local'] > 0 else None
        self.writer_discriminator = models.resnet18(num_classes=wid_num_authors) if kwargs['weight_writer_id'] > 0 else None

        embedding_module_kwargs = {'charset': charset, 'transforms': UnifontShiftTransform(gen_emb_shift)}
        self.generator = TeddyGenerator((img_height, gen_max_width), (img_height, gen_patch_width), dim=gen_dim,
                                        channels=img_channels, embedding_module=gen_emb_module, embedding_module_kwargs=embedding_module_kwargs,
                                        num_style=gen_glob_style_tokens, cnn_decoder_width=gen_cnn_decoder_width)
        
        dis_input_channels = img_channels * 2 if not single_img_dis else img_channels
        self.discriminator = HWTDiscriminator(resolution=gen_patch_width, vocab_size=len(charset) + 1, input_nc=dis_input_channels)
        self.padding_cat = PaddingCat(padding_value=1, cat_dim=1) if not single_img_dis else lambda real, fake: fake       

        self.dis_patch_sampler = UnfoldPatchSampler(dis_patch_width, img_height, dis_patch_count)
        self.style_patch_sampler = UnfoldPatchSampler(style_patch_width, img_height, style_patch_count)
        self.collector = MetricCollector()

    def generate(self, gen_texts, style_texts, style_img, enc_gen_text=None, enc_style_text=None):
        device = style_img.device

        if enc_style_text is None or enc_gen_text is None:
            enc_style_text, _ = self.text_converter.encode(style_texts, device)
            enc_gen_text, _ = self.text_converter.encode(gen_texts, device)
        fakes = self.generator(style_img, enc_style_text, enc_gen_text)
        return fakes

    def forward(self, batch):
        device = next(self.parameters()).device
        results = {}

        enc_style_text, enc_style_text_len = self.text_converter.encode(batch['style_text'], device)
        enc_gen_text, enc_gen_text_len = self.text_converter.encode(batch['gen_text'], device)

        # gen_img_len = enc_gen_text_len * self.gen_cnn_decoder_width
        gen_img_len = (batch['style_img_len'] / enc_style_text_len).floor() * enc_gen_text_len

        real_style_emb, fake_recon_emb = self.generator.forward_style(batch['style_img'], enc_style_text)
        fakes = self.generator.forward_gen(real_style_emb, enc_gen_text)

        # Adjusting the width of the fakes to be a multiple of 16
        fakes = fakes if fakes.size(-1) % 16 == 0 else F.pad(fakes, (0, 0, 8, 0), fill=1)
        
        # Reconstruction
        if batch['weight']['recon']:
            results['fakes_recon'] = self.generator.cnn_decoder(fake_recon_emb)

        # Cycle consistency
        if batch['weight']['cycle']:
            results['fake_style_emb'], _ = self.generator.forward_style(fakes, enc_gen_text)

        # Discriminator
        if batch['weight']['dis_global']:
            results['dis_glob_real_pred'] = self.discriminator(self.padding_cat(batch['style_img'], batch['same_img']))
            results['dis_glob_fake_pred'] = self.discriminator(self.padding_cat(batch['style_img'], fakes))

        if batch['weight']['dis_local']:
            real_samples = self.dis_patch_sampler(batch['style_img'], img_len=batch['style_img_len'])
            same_samples = self.dis_patch_sampler(batch['same_img'], img_len=batch['same_img_len'])
            fake_samples = self.dis_patch_sampler(fakes, img_len=gen_img_len)
            results['dis_local_real_pred'] = self.discriminator(self.padding_cat(real_samples, same_samples))
            results['dis_local_fake_pred'] = self.discriminator(self.padding_cat(real_samples, fake_samples))

        # Style, Appearence, OCR
        if batch['weight']['ocr'] or batch['weight']['style_global'] or batch['weight']['style_local'] or batch['weight']['appea_local'] or batch['weight']['writer_id']:
            fakes_rgb = repeat(fakes, 'b 1 h w -> b 3 h w')
            real_rgb = repeat(batch['style_img'], 'b 1 h w -> b 3 h w')
            other_rgb = repeat(batch['other_img'], 'b 1 h w -> b 3 h w')

        if batch['weight']['style_local'] or batch['weight']['appea_local'] or batch['weight']['writer_id']:
            fake_samples = self.style_patch_sampler(fakes_rgb, img_len=gen_img_len)
            real_samples = self.style_patch_sampler(real_rgb, img_len=batch['style_img_len'])
            other_samples = self.style_patch_sampler(other_rgb, img_len=batch['other_img_len'])

        # Style
        if batch['weight']['style_global']:
            results['style_glob_fakes'] = self.style_encoder(fakes_rgb)
            results['style_glob_positive'] = self.style_encoder(real_rgb)
            results['style_glob_negative'] = self.style_encoder(other_rgb)

        if batch['weight']['style_local']:
            results['style_local_fakes'] = self.style_encoder(fake_samples)
            results['style_local_real'] = self.style_encoder(real_samples)
            results['style_local_other'] = self.style_encoder(other_samples)

        # Appearence
        if batch['weight']['appea_local']:
            results['appea_local_fakes'] = self.apperence_encoder(fake_samples)
            results['appea_local_real'] = self.apperence_encoder(real_samples)
            results['appea_local_other'] = self.apperence_encoder(other_samples)

        # OCR
        if batch['weight']['ocr']:
            results['ocr_fake_pred'] = self.ocr(fakes_rgb)

            results['ocr_real_pred'] = None
            if 'ocr_real_train' in batch and batch['ocr_real_train']:
                results['ocr_real_pred'] = self.ocr(real_rgb)
            elif 'ocr_real_eval' in batch and batch['ocr_real_eval']:
                with torch.inference_mode():
                    results['ocr_real_pred'] = self.ocr(real_rgb)

        
        # Writer discriminator
        if batch['weight']['writer_id']:
            results['fake_global_writer_id'] = self.writer_discriminator(fakes_rgb)
            results['real_global_writer_id'] = self.writer_discriminator(real_rgb)
            results['fake_local_writer_id'] = self.writer_discriminator(fake_samples)
            results['real_local_writer_id'] = self.writer_discriminator(real_samples)

        results.update({
            'fakes': fakes,
            'real_style_emb': real_style_emb,
            'enc_gen_text': enc_gen_text,
            'enc_gen_text_len': enc_gen_text_len,
            'enc_style_text': enc_style_text,
            'enc_style_text_len': enc_style_text_len,
        })
        return results

    def generate_eval_page(self, gen_text, style_text, style_img, max_batch_size=4):
        gen_text = gen_text[:max_batch_size]
        style_text = style_text[:max_batch_size]
        style_img = style_img[:max_batch_size]
        batch_size = len(gen_text)

        alphabet = list(string.ascii_letters + string.digits)

        fakes = []
        try:
            fakes.append(self.generate(style_text, style_text, style_img))  # fakes_same_text
        except:
            pass
        fakes.append(self.generate([''.join(alphabet)] * batch_size, style_text, style_img))  # fakes_alph_join
        alph_sep = ' '.join(alphabet)
        fakes.append(self.generate([alph_sep[:len(alph_sep) // 2 + 1]] * batch_size, style_text, style_img))  # fakes_alph_sep_first_half
        fakes.append(self.generate([alph_sep[len(alph_sep) // 2:]] * batch_size, style_text, style_img))  # fakes_alph_sep_second_half
        fakes.append(self.generate(gen_text, style_text, style_img))  # fakes_1
        fakes.append(self.generate(gen_text, style_text, style_img))  # fakes_2
        fakes.append(self.generate(gen_text, style_text, style_img))  # fakes_3
        fakes.append(self.generate(gen_text, style_text, style_img))  # fakes_4
        first_word = lambda txt: txt.split()[0] if len(txt.split()) > 0 else ''
        fakes.append(self.generate([first_word(txt) for txt in gen_text], style_text, style_img))  # fakes_short
        fakes.append(self.generate([txt + ' ' + txt for txt in gen_text], style_text, style_img))  # fakes_long

        max_width = max([fake.shape[-1] for fake in fakes])
        fakes = [torch.nn.functional.pad(fake, (0, max_width - fake.shape[-1]), value=1) for fake in fakes]
        fakes = torch.cat(fakes, dim=-2)

        max_width = max([img.shape[-1] for img in style_img])
        max_height = max([fake.shape[-2] for fake in fakes])
        style_img = [torch.nn.functional.pad(img, (0, max_width - img.shape[-1], 0, max_height - img.shape[-2]), value=1) for img in style_img]
        style_img = torch.stack(style_img)

        return make_grid(torch.cat((style_img, fakes), dim=-1), nrow=1)
