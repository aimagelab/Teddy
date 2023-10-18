import torch

import pickle
from torch import nn
import numpy as np
from torchvision import models

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model.cnn_decoder import FCNDecoder
from model.ocr import OrigamiNet


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


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
        for lbl, lbl_len in zip(labels, length):
            char_list = []
            for i in range(lbl_len):
                if lbl[i] != 0 and (not (i > 0 and lbl[i - 1] == lbl[i])) and lbl[i] < len(self.charset):  # removing repeated characters and blank.
                    char_list.append(self.charset[lbl[i]])
            texts.append(''.join(char_list))
        return texts

    def _apply(self, fn):
        super(CTCLabelConverter, self)._apply(fn)
        self.device = fn(torch.empty(1)).device
        return self


class UnifontModule(nn.Module):
    def __init__(self, charset, device='cuda'):
        super(UnifontModule, self).__init__()
        self.device = device
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


class NoiseExpantion(nn.Module):
    def __init__(self, expantion_factor=1, noise_alpha=0.1):
        super().__init__()
        self.expantion_factor = expantion_factor
        self.noise_alpha = noise_alpha

    def forward(self, x):
        x = repeat(x, 'b l d -> (b e) l d', e=self.expantion_factor)
        noise = torch.randn_like(x) * self.noise_alpha
        x += noise
        return x


class TeddyGenerator(nn.Module):
    def __init__(self, image_size, patch_size, dim=512, depth=6, heads=8, mlp_dim=2048,
                 expantion_factor=1, noise_alpha=0.1, query_size=256,
                 channels=3, num_style=3, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + num_style, dim))
        self.style_tokens = nn.Parameter(torch.randn(1, num_style, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.swap_batch_len = Rearrange('b l d -> l b d')

        self.query_style_linear = torch.nn.Linear(query_size, dim)
        self.query_gen_linear = torch.nn.Linear(query_size, dim)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout)
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout)

        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=depth, enable_nested_tensor=False)  # TODO check enable nested tensor
        self.transformer_style_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=depth)
        self.transformer_gen_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=depth)

        self.batch_expantion = NoiseExpantion(expantion_factor, noise_alpha)
        self.cnn_decoder = FCNDecoder(dim=dim, out_dim=3)
        self.rearrange_expantion = Rearrange('(b e) c h w -> b e c h w', e=expantion_factor)

    def forward_style(self, style_imgs, style_tgt):
        x = self.to_patch_embedding(style_imgs)
        b, n, _ = x.shape

        style_tokens = self.style_tokens.repeat(b, 1, 1)
        x = torch.cat((style_tokens, x), dim=1)

        x += self.pos_embedding[:, :n + self.style_tokens.size(1)]
        x = self.dropout(x)

        x = self.swap_batch_len(x)
        x = self.transformer_encoder(x)

        style_tgt = self.query_style_linear(style_tgt)
        style_tgt = self.swap_batch_len(style_tgt)
        x = self.transformer_style_decoder(style_tgt, x)
        return x

    def forward_gen(self, style_emb, gen_tgt):
        gen_tgt = self.query_gen_linear(gen_tgt)
        gen_tgt = self.swap_batch_len(gen_tgt)
        x = self.transformer_gen_decoder(gen_tgt, style_emb)

        x = self.swap_batch_len(x)
        x = self.batch_expantion(x)
        x = self.cnn_decoder(x)
        x = self.rearrange_expantion(x)
        return x

    def forward(self, style_imgs, style_tgt, gen_tgt):
        style_emb = self.forward_style(style_imgs, style_tgt)
        fakes = self.forward_gen(style_emb, gen_tgt)
        return fakes


class TeddyDiscriminator(torch.nn.Module):
    def __init__(self, image_size, patch_size, dim=512, depth=6, heads=8, mlp_dim=2048, channels=3, dropout=0.1, emb_dropout=0.1, expantion_factor=1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width
        self.expantion_factor = expantion_factor
        self.patch_width = patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.single_num_patches = (image_height // patch_height) * (image_width // patch_width)
        num_patches = 2 + self.single_num_patches + self.single_num_patches  # cls_tokens + style + tgt
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_tokens = nn.Parameter(torch.randn(1, 2, dim))
        self.dropout = nn.Dropout(emb_dropout)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=depth, enable_nested_tensor=False)  # TODO check enable nested tensor
        self.fc_real_fake = nn.Linear(dim, 2)
        self.fc_same_other = nn.Linear(dim, 2)

    def forward(self, src_1_real, src_1_real_len, src_2_real, src_2_real_len, tgt_1_real, tgt_1_real_len, tgt_1_fake, fake_texts):
        # source image  target img      author  source
        # src_1_real    tgt_1_real  ->  same    real
        # src_1_real    tgt_1_fake  ->  same    fake
        # src_2_real    tgt_1_real  ->  diff    real
        # src_2_real    tgt_1_fake  ->  diff    fake
        b, *_ = src_1_real.shape
        device = src_1_real.device

        src_1_real = self.to_patch_embedding(src_1_real)
        src_2_real = self.to_patch_embedding(src_2_real)
        tgt_1_real = self.to_patch_embedding(tgt_1_real)
        tgt_1_fake = self.to_patch_embedding(tgt_1_fake)

        src_1_real_len = src_1_real_len // self.patch_width
        src_2_real_len = src_2_real_len // self.patch_width
        tgt_1_real_len = tgt_1_real_len // self.patch_width
        tgt_1_fake_len = torch.IntTensor([len(txt) for txt in fake_texts]).to(device)

        rand_idx_src_1_real = torch.randint(src_1_real_len.max() - 1, (src_1_real_len.size(0), self.single_num_patches)).to(device)
        rand_idx_src_2_real = torch.randint(src_2_real_len.max() - 1, (src_2_real_len.size(0), self.single_num_patches)).to(device)
        rand_idx_tgt_1_real = torch.randint(tgt_1_real_len.max() - 1, (tgt_1_real_len.size(0), self.single_num_patches)).to(device)
        rand_idx_tgt_1_fake = torch.randint(tgt_1_fake_len.max() - 1, (tgt_1_fake_len.size(0), self.single_num_patches)).to(device)

        rand_idx_src_1_real %= src_1_real_len.unsqueeze(-1)
        rand_idx_src_2_real %= src_2_real_len.unsqueeze(-1)
        rand_idx_tgt_1_real %= tgt_1_real_len.unsqueeze(-1)
        rand_idx_tgt_1_fake %= tgt_1_fake_len.unsqueeze(-1)

        rand_idx_src_1_real += torch.arange(rand_idx_src_1_real.size(0), device=device).unsqueeze(-1) * src_1_real.size(1)
        rand_idx_src_2_real += torch.arange(rand_idx_src_2_real.size(0), device=device).unsqueeze(-1) * src_2_real.size(1)
        rand_idx_tgt_1_real += torch.arange(rand_idx_tgt_1_real.size(0), device=device).unsqueeze(-1) * tgt_1_real.size(1)
        rand_idx_tgt_1_fake += torch.arange(rand_idx_tgt_1_fake.size(0), device=device).unsqueeze(-1) * tgt_1_fake.size(1)

        batch2flat = Rearrange('b l d -> (b l) d')
        flat2batch = Rearrange('(b l) d -> b l d', b=b)
        src_1_real = flat2batch(batch2flat(src_1_real)[rand_idx_src_1_real.flatten()])
        src_2_real = flat2batch(batch2flat(src_2_real)[rand_idx_src_2_real.flatten()])
        tgt_1_real = flat2batch(batch2flat(tgt_1_real)[rand_idx_tgt_1_real.flatten()])
        tgt_1_fake = flat2batch(batch2flat(tgt_1_fake)[rand_idx_tgt_1_fake.flatten()])

        cls_tokens = repeat(self.cls_tokens, '1 c d -> b c d', b=b*4)
        src = torch.cat((src_1_real, src_1_real, src_2_real, src_2_real), dim=0)
        tgt = torch.cat((tgt_1_real, tgt_1_fake, tgt_1_real, tgt_1_fake), dim=0)

        x = torch.cat((cls_tokens, src, tgt), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer_encoder(x)

        real_fake = self.fc_real_fake(x[:, 0])
        same_other = self.fc_same_other(x[:, 1])

        return real_fake, same_other


class Teddy(torch.nn.Module):
    def __init__(self, charset, dim=512, img_height=32, style_max_width=2512, patch_width=16, expantion_factor=4, discriminator_width=31 * 16) -> None:
        super().__init__()
        self.expantion_factor = expantion_factor
        self.unifont_embedding = UnifontModule(charset)
        self.text_converter = CTCLabelConverter(charset)
        self.ocr = OrigamiNet(o_classes=len(self.unifont_embedding))
        freeze(self.ocr)

        self.generator = TeddyGenerator((img_height, style_max_width), (img_height, patch_width), dim=dim,
                                        expantion_factor=expantion_factor, query_size=self.unifont_embedding.symbols_size)
        self.discriminator = TeddyDiscriminator((img_height, discriminator_width), (img_height, patch_width), dim=dim,
                                                expantion_factor=expantion_factor)

    def forward(self, style_imgs, style_imgs_len, style_text, gen_text, same_author_imgs, same_author_imgs_len, other_author_imgs, other_author_imgs_len):
        enc_style_text, _ = self.text_converter.encode(style_text)
        enc_gen_text, _ = self.text_converter.encode(gen_text)

        style_tgt = self.unifont_embedding(enc_style_text)
        gen_tgt = self.unifont_embedding(enc_gen_text)

        src_style_emb = self.generator.forward_style(style_imgs, style_tgt)
        fakes = self.generator.forward_gen(src_style_emb, gen_tgt)

        real_fake_pred, same_other_pred = self.discriminator(
            style_imgs, style_imgs_len,
            same_author_imgs, same_author_imgs_len,
            other_author_imgs, other_author_imgs_len,
            fakes[:, 0], gen_text  # Take only the first image of each expansion
        )

        fakes_exp = rearrange(fakes, 'b e c h w -> (b e) c h w')
        text_pred = self.ocr(fakes_exp)
        text_pred = rearrange(text_pred, '(b e) l c -> b e l c', e=self.expantion_factor)

        gen_tgt = repeat(gen_tgt, 'b l d -> (b e) l d', e=self.expantion_factor)
        gen_style_emb = self.generator.forward_style(fakes_exp, gen_tgt)
        return fakes, real_fake_pred, same_other_pred, text_pred, src_style_emb, gen_style_emb
