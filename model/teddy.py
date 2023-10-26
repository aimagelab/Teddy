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
    def __init__(self, image_size, patch_size, dim=512, depth=6, heads=8, mlp_dim=512,
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
        self.dropout = nn.Dropout(emb_dropout)

        self.query_style_linear = torch.nn.Linear(query_size, dim)
        self.query_gen_linear = torch.nn.Linear(query_size, dim)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True, norm_first=True)
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True, norm_first=True)

        encoder_norm = nn.LayerNorm(dim)
        decoder_norm = nn.LayerNorm(dim)

        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=depth, norm=encoder_norm)
        self.transformer_style_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=depth, norm=decoder_norm)
        self.transformer_gen_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=depth, norm=decoder_norm)

        self.batch_expantion = NoiseExpantion(expansion_factor, noise_alpha)
        self.cnn_decoder = FCNDecoder(dim=dim, out_dim=channels)
        self.rearrange_expantion = Rearrange('(b e) c h w -> b e c h w', e=expansion_factor)

    def forward_style(self, style_imgs, style_tgt):
        x = self.to_patch_embedding(style_imgs)
        b, n, _ = x.shape

        style_tokens = self.style_tokens.repeat(b, 1, 1)
        x = torch.cat((style_tokens, x), dim=1)

        x += self.pos_embedding[:, :n + self.style_tokens.size(1)]
        x = self.dropout(x)

        x = self.transformer_encoder(x)

        style_tgt = self.query_style_linear(style_tgt)
        x = self.transformer_style_decoder(style_tgt, x)
        return x

    def forward_gen(self, style_emb, gen_tgt):
        gen_tgt = self.query_gen_linear(gen_tgt)
        x = self.transformer_gen_decoder(gen_tgt, style_emb)

        x = self.batch_expantion(x)
        x = self.cnn_decoder(x)
        x = self.rearrange_expantion(x)
        return x

    def forward(self, style_imgs, style_tgt, gen_tgt):
        style_emb = self.forward_style(style_imgs, style_tgt)
        fakes = self.forward_gen(style_emb, gen_tgt)
        return fakes


class TeddyDiscriminator(torch.nn.Module):
    def __init__(self, image_size, patch_size, dim=512, depth=6, heads=8, mlp_dim=2048, channels=3, dropout=0.1, emb_dropout=0.1, expansion_factor=1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width
        self.expansion_factor = expansion_factor
        self.patch_width = patch_width

        self.to_patch_sequence = Rearrange('b c h (p pw) -> b p (h pw c)', pw=self.patch_width)
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.single_num_patches = (image_height // patch_height) * (image_width // patch_width)
        num_patches = 2 + self.single_num_patches + self.single_num_patches  # cls_tokens + style + tgt
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_tokens = nn.Parameter(torch.randn(1, 2, dim))
        self.dropout = nn.Dropout(emb_dropout)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=depth)
        self.fc_real_fake = nn.Linear(dim, 1)
        self.fc_same_other = nn.Linear(dim, 1)

    def old_forward(self, src_1_real, src_1_real_len, src_2_real, src_2_real_len, tgt_1_real, tgt_1_real_len, tgt_1_fake, fake_texts):
        # source image  target img      author  source
        # src_1_real    tgt_1_real  ->  same    real
        # src_1_real    tgt_1_fake  ->  same    fake
        # src_2_real    tgt_1_real  ->  diff    real
        # src_2_real    tgt_1_fake  ->  diff    fake
        b, *_ = src_1_real.shape
        device = src_1_real.device

        # src_1_real = src_1_real.transpose(2, 3).reshape(b, c, src_1_real.size(2) // self.patch_width, -1)[:, :, 0, :].reshape(4, 1, 16, 32).transpose(2, 3)
        # steps = src_1_real.size(-1) // self.patch_width
        # fake = torch.cat([torch.ones((1, 32, 16), device=device) * i / steps for i in range(steps)], dim=-1)
        # src_1_real = torch.stack([fake] * b)
        src_1_real = self.to_patch_sequence(src_1_real)
        src_2_real = self.to_patch_sequence(src_2_real)
        tgt_1_real = self.to_patch_sequence(tgt_1_real)
        tgt_1_fake = self.to_patch_sequence(tgt_1_fake)

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

        rand_idx_src_1_real = rand_idx_src_1_real.flatten()
        rand_idx_src_2_real = rand_idx_src_2_real.flatten()
        rand_idx_tgt_1_real = rand_idx_tgt_1_real.flatten()
        rand_idx_tgt_1_fake = rand_idx_tgt_1_fake.flatten()

        batch2flat = Rearrange('b l d -> (b l) d')
        flat2batch = Rearrange('(b l) d -> b l d', b=b)
        src_1_real = flat2batch(batch2flat(src_1_real)[rand_idx_src_1_real])
        src_2_real = flat2batch(batch2flat(src_2_real)[rand_idx_src_2_real])
        tgt_1_real = flat2batch(batch2flat(tgt_1_real)[rand_idx_tgt_1_real])
        tgt_1_fake = flat2batch(batch2flat(tgt_1_fake)[rand_idx_tgt_1_fake])

        # save_image(src_1_real.reshape(b, 1, -1, 32).transpose(2, 3), 'src_1_real.png')
        # save_image(src_2_real.reshape(b, 1, -1, 32).transpose(2, 3), 'src_2_real.png')
        # save_image(tgt_1_real.reshape(b, 1, -1, 32).transpose(2, 3), 'tgt_1_real.png')
        # save_image(tgt_1_fake.reshape(b, 1, -1, 32).transpose(2, 3), 'tgt_1_fake.png')

        src_1_real = self.to_patch_embedding(src_1_real)
        src_2_real = self.to_patch_embedding(src_2_real)
        tgt_1_real = self.to_patch_embedding(tgt_1_real)
        tgt_1_fake = self.to_patch_embedding(tgt_1_fake)

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

    def forward(self, src, src_len, tgt, tgt_len):
        b, *_ = src.shape
        device = src.device

        src = self.to_patch_sequence(src)
        tgt = self.to_patch_sequence(tgt)

        src_len = src_len // self.patch_width
        tgt_len = tgt_len // self.patch_width

        src_rand_idx = torch.randint(src_len.max() - 1, (src_len.size(0), self.single_num_patches)).to(device)
        tgt_rand_idx = torch.randint(tgt_len.max() - 1, (tgt_len.size(0), self.single_num_patches)).to(device)

        src_rand_idx %= src_len.unsqueeze(-1)
        tgt_rand_idx %= tgt_len.unsqueeze(-1)

        src_rand_idx += torch.arange(src_rand_idx.size(0), device=device).unsqueeze(-1) * src_1_real.size(1)
        tgt_rand_idx += torch.arange(tgt_rand_idx.size(0), device=device).unsqueeze(-1) * tgt_1_real.size(1)

        src_rand_idx = src_rand_idx.flatten()
        tgt_rand_idx = tgt_rand_idx.flatten()

        batch2flat = Rearrange('b l d -> (b l) d')
        flat2batch = Rearrange('(b l) d -> b l d', b=b)
        src = flat2batch(batch2flat(src)[src_rand_idx])
        tgt = flat2batch(batch2flat(tgt)[tgt_rand_idx])

        # save_image(src.reshape(b, 1, -1, 32).transpose(2, 3), 'src.png')
        # save_image(tgt.reshape(b, 1, -1, 32).transpose(2, 3), 'tgt.png')

        src = self.to_patch_embedding(src)
        tgt = self.to_patch_embedding(tgt)

        cls_tokens = repeat(self.cls_tokens, '1 c d -> b c d', b=b)

        x = torch.cat((cls_tokens, src, tgt), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer_encoder(x)

        real_fake = self.fc_real_fake(x[:, 0])
        same_other = self.fc_same_other(x[:, 1])

        return real_fake, same_other


class ResnetDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(512, 1)

    def forward(self, src):
        return self.resnet(src)


class Teddy(torch.nn.Module):
    def __init__(self, charset, dim=512, img_height=32, style_max_width=2512, patch_width=16, expansion_factor=4, discriminator_width=5 * 16, img_channels=3) -> None:
        super().__init__()
        self.expansion_factor = expansion_factor
        self.unifont_embedding = UnifontModule(charset)
        self.text_converter = CTCLabelConverter(charset)
        self.ocr = OrigamiNet(o_classes=len(charset) + 1)

        self.generator = TeddyGenerator((img_height, style_max_width), (img_height, patch_width), dim=dim, expansion_factor=expansion_factor,
                                        query_size=self.unifont_embedding.symbols_size, channels=img_channels)
        # self.discriminator = TeddyDiscriminator((img_height, discriminator_width), (img_height, patch_width), dim=dim,
        #                                         expansion_factor=expansion_factor, channels=img_channels)
        self.discriminator = ResnetDiscriminator()

    def forward(self, batch):
        # style_imgs, style_imgs_len, style_text, gen_text, same_author_imgs, same_author_imgs_len, other_author_imgs, other_author_imgs_len
        enc_style_text, enc_style_text_len = self.text_converter.encode(batch['style_texts'])
        enc_gen_texts, enc_gen_texts_len = self.text_converter.encode(batch['gen_texts'])

        style_tgt = self.unifont_embedding(enc_style_text)
        gen_tgt = self.unifont_embedding(enc_gen_texts)

        src_style_emb = self.generator.forward_style(batch['style_imgs'], style_tgt)
        fakes = self.generator.forward_gen(src_style_emb, gen_tgt)

        #############################
        # Teddy discriminator

        # real_fake_pred, same_other_pred = self.discriminator(
        #     batch['style_imgs'], batch['style_imgs_len'].int(),
        #     batch['same_author_imgs'], batch['same_author_imgs_len'].int(),
        #     batch['other_author_imgs'], batch['other_author_imgs_len'].int(),
        #     fakes[:, 0], batch['gen_texts']  # Take only the first image of each expansion
        # )

        # source image  target img      author  source
        # src_1_real    tgt_1_real  ->  same    real
        # src_2_real    tgt_1_real  ->  diff    real
        # rf_pred, so_pred = self.discriminator(
        #     cat_pad(batch['style_imgs'], batch['same_author_imgs']), cat_pad(batch['style_imgs_len'], batch['same_author_imgs_len']),
        #     cat_pad(batch['other_author_imgs'], batch['same_author_imgs']), cat_pad(batch['other_author_imgs_len'], batch['same_author_imgs_len']),
        # )

        # source image  target img      author  source
        # src_1_real    tgt_1_fake  ->  same    fake
        # src_2_real    tgt_1_fake  ->  diff    fake
        #############################

        dis_real_pred = self.discriminator(batch['style_imgs'])
        dis_fake_pred = self.discriminator(fakes[:, 0])

        # fakes_rgb = repeat(fakes, 'b e 1 h w -> (b e) 3 h w')
        # real_rgb = repeat(batch['style_imgs'], 'b 1 h w -> b 3 h w')
        # enc_gen_texts = repeat(enc_gen_texts, 'b w -> (b e) w', e=self.expansion_factor)
        # enc_gen_texts_len = repeat(enc_gen_texts_len, 'b -> (b e)', e=self.expansion_factor)

        # freeze(self.ocr)
        # fake_text_pred = self.ocr(fakes_rgb)
        # unfreeze(self.ocr)
        # real_text_pred = self.ocr(real_rgb)

        # fakes_exp = rearrange(fakes, 'b e c h w -> (b e) c h w')
        # gen_tgt = repeat(gen_tgt, 'b l d -> (b e) l d', e=self.expansion_factor)
        # gen_style_emb = self.generator.forward_style(fakes_exp, gen_tgt)

        results = {
            'fakes': fakes,
            'real_fake_pred': torch.cat((dis_real_pred, dis_fake_pred), dim=0),
            # 'same_other_pred': same_other_pred,
            # 'fake_text_pred': fake_text_pred,
            # 'real_text_pred': real_text_pred,
            # 'src_style_emb': src_style_emb,
            # 'gen_style_emb': gen_style_emb,
            # 'enc_gen_texts': enc_gen_texts,
            # 'enc_gen_texts_len': enc_gen_texts_len,
            # 'enc_style_text': enc_style_text,
            # 'enc_style_text_len': enc_style_text_len,
        }
        return results
