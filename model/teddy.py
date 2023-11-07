from typing import Any
import torch
import math

import pickle
from torch import nn
import numpy as np
from torchvision import models

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model.cnn_decoder import FCNDecoder
from model.ocr import OrigamiNet
from model.hwt.model import Discriminator as HWTDiscriminator
from util.functional import Clock, MetricCollector


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
        self.device = 'cpu'
        self.charset = sorted(set(charset))

        # NOTE: 0 is reserved for 'blank' token required by CTCLoss
        self.dict = {char: i + 1 for i, char in enumerate(self.charset)}
        self.charset.insert(0, '[blank]')  # dummy '[blank]' token for CTCLoss (index 0)

    def encode(self, labels):
        assert set(''.join(labels)) < set(self.charset), f'The following character are not in charset {set("".join(labels)) - set(self.charset)}'
        length = torch.LongTensor([len(lbl) for lbl in labels])
        labels = [torch.LongTensor([self.dict[char] for char in lbl]) for lbl in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
        return labels.to(self.device), length.to(self.device)

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

    def _apply(self, fn):
        super(CTCLabelConverter, self)._apply(fn)
        self.device = fn(torch.empty(1)).device
        return self


class UnifontModule(nn.Module):
    def __init__(self, charset):
        super(UnifontModule, self).__init__()
        self.charset = set(charset)
        self.symbols = self.get_symbols()
        self.symbols_size = self.symbols.size(1)

    def get_symbols(self):
        with open(f"files/unifont.pickle", "rb") as f:
            symbols = pickle.load(f)

        symbols = {sym['idx'][0]: sym['mat'].astype(np.float32).flatten() for sym in symbols}
        symbols = [symbols[ord(char)] for char in self.charset]
        symbols.insert(0, np.zeros_like(symbols[0]))
        symbols = np.stack(symbols)
        return torch.from_numpy(symbols).float()

    def _apply(self, fn):
        super(UnifontModule, self)._apply(fn)
        self.symbols = fn(self.symbols)
        return self

    def forward(self, QR):
        return self.symbols[QR]

    def __len__(self):
        return len(self.symbols)


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
                 expansion_factor=1, noise_alpha=0.0, query_size=256,
                 channels=3, num_style=3, dropout=0.1, emb_dropout=0.1):
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

        self.query_style_linear = torch.nn.Linear(query_size, dim)
        self.query_gen_linear = torch.nn.Linear(query_size, dim)

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

        style_tokens = self.style_tokens.repeat(b, 1, 1)
        x = torch.cat((style_tokens, x), dim=1)
        x += self.pos_embedding[:, :n + self.style_tokens.size(1)]

        x = self.transformer_encoder(x)

        style_tgt = self.query_style_linear(style_tgt)
        x = self.transformer_style_decoder(style_tgt, x)
        return x

    def forward_gen(self, style_emb, gen_tgt):
        gen_tgt = self.query_gen_linear(gen_tgt)
        x = self.transformer_gen_decoder(gen_tgt, style_emb)

        x = self.batch_expansion(x)
        x = self.cnn_decoder(x)
        x = self.rearrange_expansion(x)
        return x

    def forward(self, style_imgs, style_tgt, gen_tgt):
        style_emb = self.forward_style(style_imgs, style_tgt)
        fakes = self.forward_gen(style_emb, gen_tgt)
        return fakes


# class TeddyDiscriminator(torch.nn.Module):
#     def __init__(self, image_size, patch_size, dim=512, depth=3, heads=8, mlp_dim=2048, channels=3, dropout=0.1, emb_dropout=0.1, expansion_factor=1):
#         super().__init__()
#         image_height, image_width = pair(image_size)
#         patch_height, patch_width = pair(patch_size)

#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

#         patch_dim = channels * patch_height * patch_width
#         self.expansion_factor = expansion_factor
#         self.patch_width = patch_width

#         self.to_patch_sequence = Rearrange('b c h (p pw) -> b p (h pw c)', pw=self.patch_width)
#         self.to_patch_embedding = nn.Sequential(
#             nn.LayerNorm(patch_dim),
#             nn.Linear(patch_dim, dim),
#             nn.LayerNorm(dim),
#         )

#         self.num_patches = (image_height // patch_height) * (image_width // patch_width)
#         self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
#         self.cls_tokens = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=depth)
#         self.fc_real_fake = nn.Linear(dim, 1)

#     def forward(self, src):
#         b, *_ = src.shape
#         src = self.to_patch_sequence(src)
#         src = self.to_patch_embedding(src)

#         cls_tokens = repeat(self.cls_tokens, '1 c d -> b c d', b=b)

#         x = torch.cat((cls_tokens, src), dim=1)
#         x += self.pos_embedding
#         x = self.dropout(x)

#         x = self.transformer_encoder(x)

#         real_fake = self.fc_real_fake(x[:, 0])

#         return real_fake


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

    def __call__(self, img):
        b, c, h, w = img.shape
        device = img.device
        img_len = torch.tensor([w] * b, device=device)
        img_len = img_len // self.unit
        img_seq = self.img_to_seq(img[:, :, :, :img_len.max() * self.unit])
        rand_idx = torch.randint(img_len.max() - 1 - (self.patch_width // self.unit), (b, self.patch_count)).to(device)
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
                 gen_expansion_factor, dis_patch_width, dis_patch_count, style_patch_width, style_patch_count, **kwargs) -> None:
        super().__init__()
        self.expansion_factor = gen_expansion_factor
        self.unifont_embedding = UnifontModule(charset)
        self.text_converter = CTCLabelConverter(charset)
        self.ocr = OrigamiNet(o_classes=len(charset) + 1)
        self.style_encoder = FontSquareEncoder()
        freeze(self.style_encoder)
        self.generator = TeddyGenerator((img_height, gen_max_width), (img_height, gen_patch_width), dim=gen_dim, expansion_factor=gen_expansion_factor,
                                        query_size=self.unifont_embedding.symbols_size, channels=img_channels)
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

    def forward(self, batch):
        enc_style_text, enc_style_text_len = self.text_converter.encode(batch['style_texts'])
        enc_gen_text, enc_gen_text_len = self.text_converter.encode(batch['gen_texts'])

        device = self.generator.query_gen_linear.weight.device

        style_tgt = self.unifont_embedding(enc_style_text).to(device)
        gen_tgt = self.unifont_embedding(enc_gen_text).to(device)

        src_style_emb = self.generator.forward_style(batch['style_imgs'], style_tgt)
        fakes = self.generator.forward_gen(src_style_emb, gen_tgt)

        dis_glob_real_pred = self.discriminator(batch['style_imgs'])
        dis_glob_fake_pred = self.discriminator(fakes[:, 0])

        real = self.dis_patch_sampler(batch['style_imgs'])
        fake = self.dis_patch_sampler(fakes[:, 0])
        dis_local_real_pred = self.discriminator(real)
        dis_local_fake_pred = self.discriminator(fake)

        fakes_rgb = repeat(fakes, 'b e 1 h w -> (b e) 3 h w')
        real_rgb = repeat(batch['style_imgs'], 'b 1 h w -> b 3 h w')
        other_rgb = repeat(batch['other_author_imgs'], 'b 1 h w -> b 3 h w')
        enc_gen_text = repeat(enc_gen_text, 'b w -> (b e) w', e=self.expansion_factor)
        enc_gen_text_len = repeat(enc_gen_text_len, 'b -> (b e)', e=self.expansion_factor)

        real_samples = self.style_patch_sampler(real_rgb)
        style_local_real = self.style_encoder(real_samples)
        fakes_samples = self.style_patch_sampler(fakes_rgb)
        style_local_fakes = self.style_encoder(fakes_samples)
        style_glob_fakes = self.style_encoder(fakes_rgb)
        style_glob_negative = self.style_encoder(other_rgb)
        style_glob_positive = self.style_encoder(real_rgb)

        ocr_fake_pred = self.ocr(fakes_rgb)

        if 'last_batch' in batch and batch['last_batch']:
            with torch.inference_mode():
                ocr_real_pred = self.ocr(real_rgb)
        else:
            ocr_real_pred = None

        # fakes_exp = rearrange(fakes, 'b e c h w -> (b e) c h w')
        # gen_tgt = repeat(gen_tgt, 'b l d -> (b e) l d', e=self.expansion_factor)
        # gen_style_emb = self.generator.forward_style(fakes_exp, gen_tgt)

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
            'style_glob_fakes': style_glob_fakes,
            'style_glob_negative': style_glob_negative,
            'style_glob_positive': style_glob_positive,
        }
        return results
