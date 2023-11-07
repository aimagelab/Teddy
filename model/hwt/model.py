import torch
import pandas as pd
from model.hwt.OCR_network import *
from torch.nn import CTCLoss, MSELoss, L1Loss
from torch.nn.utils import clip_grad_norm_
import random
import unicodedata
import sys
import torchvision.models as models
from model.hwt.transformer import *
from model.hwt.BigGAN_networks import *
from model.hwt.OCR_network import *
from model.hwt.blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock
from model.hwt.inception import InceptionV3, calculate_frechet_distance
import time
import matplotlib.pyplot as plt
import shutil
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def get_rgb(x):
    R = 255 - int(int(x > 0.5)*255*(x-0.5)/0.5)
    G = 0
    B = 255 + int(int(x < 0.5)*255*(x-0.5)/0.5)
    return R, G, B


def get_page_from_words(word_lists, MAX_IMG_WIDTH=800):

    line_all = []
    line_t = []

    width_t = 0

    for i in word_lists:

        width_t = width_t + i.shape[1] + 16

        if width_t > MAX_IMG_WIDTH:

            line_all.append(np.concatenate(line_t, 1))

            line_t = []

            width_t = i.shape[1] + 16

        line_t.append(i)
        line_t.append(np.ones((i.shape[0], 16)))

    if len(line_all) == 0:

        line_all.append(np.concatenate(line_t, 1))

    max_lin_widths = MAX_IMG_WIDTH  # max([i.shape[1] for i in line_all])
    gap_h = np.ones([16, max_lin_widths])

    page_ = []

    for l in line_all:

        pad_ = np.ones([l.shape[0], max_lin_widths - l.shape[1]])

        page_.append(np.concatenate([l, pad_], 1))
        page_.append(gap_h)

    page = np.concatenate(page_, 0)

    return page*255


class FCNDecoder(nn.Module):
    def __init__(self, ups=3, n_res=2, dim=512, out_dim=1, res_norm='in', activ='relu', pad_type='reflect'):
        super(FCNDecoder, self).__init__()

        self.fc = nn.Linear(dim, dim * 8)

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm,
                                 activ, pad_type=pad_type)]
        for i in range(ups):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm='in',
                                       activation=activ,
                                       pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        # y = self.model(x)
        # return y
        x = self.fc(x)
        b, c, _ = x.shape
        x = x.view(b, c * 2, 4, -1)
        x = x.permute(0, 3, 2, 1)
        return self.model(x)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        INP_CHANNEL = NUM_EXAMPLES
        if IS_SEQ:
            INP_CHANNEL = 1

        encoder_layer = TransformerEncoderLayer(TN_HIDDEN_DIM, TN_NHEADS, TN_DIM_FEEDFORWARD,
                                                TN_DROPOUT, "relu", True)
        encoder_norm = nn.LayerNorm(TN_HIDDEN_DIM) if True else None
        self.encoder = TransformerEncoder(encoder_layer, TN_ENC_LAYERS, encoder_norm)

        decoder_layer = TransformerDecoderLayer(TN_HIDDEN_DIM, TN_NHEADS, TN_DIM_FEEDFORWARD,
                                                TN_DROPOUT, "relu", True)
        decoder_norm = nn.LayerNorm(TN_HIDDEN_DIM)
        self.decoder = TransformerDecoder(decoder_layer, TN_DEC_LAYERS, decoder_norm,
                                          return_intermediate=True)

        self.Feat_Encoder = nn.Sequential(*([nn.Conv2d(INP_CHANNEL, 64, kernel_size=7, stride=2, padding=3, bias=False)] + list(models.resnet18(pretrained=True).children())[1:-2]))

        self.query_embed = nn.Embedding(VOCAB_SIZE, TN_HIDDEN_DIM)

        self.linear_q = nn.Linear(TN_DIM_FEEDFORWARD, TN_DIM_FEEDFORWARD*8)

        self.DEC = FCNDecoder(res_norm='in')

        self._muE = nn.Linear(512, 512)
        self._logvarE = nn.Linear(512, 512)

        self._muD = nn.Linear(512, 512)
        self._logvarD = nn.Linear(512, 512)

        self.l1loss = nn.L1Loss()

        self.noise = torch.distributions.Normal(loc=torch.tensor([0.]), scale=torch.tensor([1.0]))

    def reparameterize(self, mu, logvar):

        mu = torch.unbind(mu, 1)
        logvar = torch.unbind(logvar, 1)

        outs = []

        for m, l in zip(mu, logvar):

            sigma = torch.exp(l)
            eps = torch.cuda.FloatTensor(l.size()[0], 1).normal_(0, 1)
            eps = eps.expand(sigma.size())

            out = m + sigma*eps

            outs.append(out)

        return torch.stack(outs, 1)

    def Eval(self, ST, QRS):

        if IS_SEQ:
            B, N, R, C = ST.shape
            FEAT_ST = self.Feat_Encoder(ST.view(B*N, 1, R, C))
            FEAT_ST = FEAT_ST.view(B, 512, 1, -1)
        else:
            FEAT_ST = self.Feat_Encoder(ST)

        FEAT_ST_ENC = FEAT_ST.flatten(2).permute(2, 0, 1)

        memory = self.encoder(FEAT_ST_ENC)

        if IS_KLD:

            Ex = memory.permute(1, 0, 2)

            memory_mu = self._muE(Ex)
            memory_logvar = self._logvarE(Ex)

            memory = self.reparameterize(memory_mu, memory_logvar).permute(1, 0, 2)

        OUT_IMGS = []

        for i in range(QRS.shape[1]):

            QR = QRS[:, i, :]

            if ALL_CHARS:
                QR_EMB = self.query_embed.weight.repeat(batch_size, 1, 1).permute(1, 0, 2)
            else:
                QR_EMB = self.query_embed.weight[QR].permute(1, 0, 2)

            tgt = torch.zeros_like(QR_EMB)

            hs = self.decoder(tgt, memory, query_pos=QR_EMB)

            if IS_KLD:

                Dx = hs[0].permute(1, 0, 2)

                hs_mu = self._muD(Dx)
                hs_logvar = self._logvarD(Dx)

                hs = self.reparameterize(hs_mu, hs_logvar).permute(1, 0, 2).unsqueeze(0)

            h = hs.transpose(1, 2)[-1]  # torch.cat([hs.transpose(1, 2)[-1], QR_EMB.permute(1,0,2)], -1)
            if ADD_NOISE:
                h = h + self.noise.sample(h.size()).squeeze(-1).to(DEVICE)

            h = self.linear_q(h)
            h = h.contiguous()

            if ALL_CHARS:
                h = torch.stack([h[i][QR[i]] for i in range(batch_size)], 0)

            h = h.view(h.size(0), h.shape[1]*2, 4, -1)
            h = h.permute(0, 3, 2, 1)

            h = self.DEC(h)

            OUT_IMGS.append(h.detach())

        return OUT_IMGS

    def forward(self, ST, QR, QRs=None, mode='train'):

        # Attention Visualization Init

        enc_attn_weights, dec_attn_weights = [], []

        self.hooks = [

            self.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            self.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]

        # Attention Visualization Init

        B, N, R, C = ST.shape
        FEAT_ST = self.Feat_Encoder(ST.view(B*N, 1, R, C))
        FEAT_ST = FEAT_ST.view(B, 512, 1, -1)

        FEAT_ST_ENC = FEAT_ST.flatten(2).permute(2, 0, 1)

        memory = self.encoder(FEAT_ST_ENC)

        QR_EMB = self.query_embed.weight[QR].permute(1, 0, 2)

        tgt = torch.zeros_like(QR_EMB)

        hs = self.decoder(tgt, memory, query_pos=QR_EMB)

        h = hs.transpose(1, 2)[-1]  # torch.cat([hs.transpose(1, 2)[-1], QR_EMB.permute(1,0,2)], -1)

        if ADD_NOISE:
            h = h + self.noise.sample(h.size()).squeeze(-1).to(DEVICE)

        h = self.linear_q(h)
        h = h.contiguous()

        h = h.view(h.size(0), h.shape[1]*2, 4, -1)
        h = h.permute(0, 3, 2, 1)

        h = self.DEC(h)

        self.dec_attn_weights = dec_attn_weights[-1].detach()
        self.enc_attn_weights = enc_attn_weights[-1].detach()

        for hook in self.hooks:
            hook.remove()

        return h


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class TeddyGenerator(nn.Module):
    def __init__(self, vocab_size, image_size=(32, 2512), patch_size=(32, 16), dim=512, depth=3, heads=8, mlp_dim=512,
                 expansion_factor=1, noise_alpha=0.0, query_size=256,
                 channels=1, num_style=3, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.patch_width = patch_width

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b h (p pw) -> b p (h pw)', pw=self.patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + num_style, dim))
        self.style_tokens = nn.Parameter(torch.randn(1, num_style, dim))
        self.query_style_embed = nn.Embedding(vocab_size, dim)
        self.query_gen_embed = nn.Embedding(vocab_size, dim)
        # self.query_style_linear = torch.nn.Linear(query_size, dim)
        # self.query_gen_linear = torch.nn.Linear(query_size, dim)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)

        encoder_norm = nn.LayerNorm(dim)
        decoder_norm = nn.LayerNorm(dim)

        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=depth, norm=encoder_norm)
        self.transformer_style_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=depth, norm=decoder_norm)
        self.transformer_gen_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=depth, norm=decoder_norm)

        # self.batch_expansion = NoiseExpansion(expansion_factor, noise_alpha)
        self.cnn_decoder = FCNDecoder(dim=dim, out_dim=channels)
        # self.rearrange_expansion = Rearrange('(b e) c h w -> b e c h w', e=expansion_factor)

    def forward_style(self, style_imgs, style_tgt):
        x = self.to_patch_embedding(style_imgs)
        b, n, _ = x.shape

        style_tokens = self.style_tokens.repeat(b, 1, 1)
        x = torch.cat((style_tokens, x), dim=1)
        x += self.pos_embedding[:, :n + self.style_tokens.size(1)]

        x = self.transformer_encoder(x)

        style_tgt = self.query_style_embed.weight[style_tgt]
        x = self.transformer_style_decoder(style_tgt, x)
        return x

    def forward_gen(self, style_emb, gen_tgt):
        gen_tgt = self.query_gen_embed.weight[gen_tgt]
        x = self.transformer_gen_decoder(gen_tgt, style_emb)

        # x = self.batch_expansion(x)
        x = self.cnn_decoder(x)
        # x = self.rearrange_expansion(x)
        return x

    # def forward(self, style_imgs, style_tgt, gen_tgt):
    #     style_emb = self.forward_style(style_imgs, style_tgt)
    #     fakes = self.forward_gen(style_emb, gen_tgt)
    #     return fakes

    def forward(self, style_imgs, style_tgt, gen_tgt):
        style_emb = self.forward_style(style_imgs, style_tgt)
        fakes = self.forward_gen(style_emb, gen_tgt)
        return fakes


class TRGAN(nn.Module):

    def __init__(self):
        super(TRGAN, self).__init__()

        self.epsilon = 1e-7
        # self.netG = Generator().to(DEVICE)
        self.netG = nn.DataParallel(TeddyGenerator(VOCAB_SIZE)).to(DEVICE)
        self.netD = nn.DataParallel(Discriminator()).to(DEVICE)
        self.netW = nn.DataParallel(WDiscriminator()).to(DEVICE)
        self.netconverter = strLabelConverter(ALPHABET)
        self.netOCR = CRNN().to(DEVICE)
        self.OCR_criterion = CTCLoss(zero_infinity=True, reduction='none')

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception = InceptionV3([block_idx]).to(DEVICE)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=G_LR, betas=(0.0, 0.999), weight_decay=0, eps=1e-8)
        self.optimizer_OCR = torch.optim.Adam(self.netOCR.parameters(),
                                              lr=OCR_LR, betas=(0.0, 0.999), weight_decay=0,
                                              eps=1e-8)

        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=D_LR, betas=(0.0, 0.999), weight_decay=0, eps=1e-8)

        self.optimizer_wl = torch.optim.Adam(self.netW.parameters(),
                                             lr=W_LR, betas=(0.0, 0.999), weight_decay=0, eps=1e-8)

        self.optimizers = [self.optimizer_G, self.optimizer_OCR, self.optimizer_D, self.optimizer_wl]

        self.optimizer_G.zero_grad()
        self.optimizer_OCR.zero_grad()
        self.optimizer_D.zero_grad()
        self.optimizer_wl.zero_grad()

        self.loss_G = 0
        self.loss_D = 0
        self.loss_Dfake = 0
        self.loss_Dreal = 0
        self.loss_OCR_fake = 0
        self.loss_OCR_real = 0
        self.loss_w_fake = 0
        self.loss_w_real = 0
        self.Lcycle1 = 0
        self.Lcycle2 = 0
        self.lda1 = 0
        self.lda2 = 0
        self.KLD = 0

        with open(ENGLISH_WORDS_PATH, 'rb') as f:
            self.lex = f.read().splitlines()
        lex = []
        for word in self.lex:
            try:
                word = word.decode("utf-8")
            except:
                continue
            if len(word) < 20:
                lex.append(word)
        self.lex = lex

        f = open('mytext.txt', 'r')

        self.text = [j.encode() for j in sum([i.split(' ') for i in f.readlines()], [])]  # [:NUM_EXAMPLES]
        self.eval_text_encode, self.eval_len_text = self.netconverter.encode(self.text)
        self.eval_text_encode = self.eval_text_encode.to(DEVICE).repeat(batch_size, 1, 1)

    def save_images_for_fid_calculation(self, dataloader, epoch, mode='train'):

        self.real_base = os.path.join('saved_images', EXP_NAME, 'Real')
        self.fake_base = os.path.join('saved_images', EXP_NAME, 'Fake')

        if os.path.isdir(self.real_base):
            shutil.rmtree(self.real_base)
        if os.path.isdir(self.fake_base):
            shutil.rmtree(self.fake_base)

        os.mkdir(self.real_base)
        os.mkdir(self.fake_base)

        for step, data in enumerate(dataloader):

            ST = data['simg'].cuda()
            self.fakes = self.netG.Eval(ST, self.eval_text_encode)
            fake_images = torch.cat(self.fakes, 1).detach().cpu().numpy()

            for i in range(fake_images.shape[0]):
                for j in range(fake_images.shape[1]):
                    # cv2.imwrite(os.path.join(self.real_base, str(step*batch_size + i)+'_'+str(j)+'.png'), 255*(real_images[i,j]))
                    cv2.imwrite(os.path.join(self.fake_base, str(step*batch_size + i)+'_'+str(j)+'.png'), 255*(fake_images[i, j]))

        if mode == 'train':

            TextDatasetObj = TextDataset(num_examples=self.eval_text_encode.shape[1])
            dataset_real = torch.utils.data.DataLoader(
                TextDatasetObj,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True, drop_last=True,
                collate_fn=TextDatasetObj.collate_fn)

        elif mode == 'test':

            TextDatasetObjval = TextDatasetval(num_examples=self.eval_text_encode.shape[1])
            dataset_real = torch.utils.data.DataLoader(
                TextDatasetObjval,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True, drop_last=True,
                collate_fn=TextDatasetObjval.collate_fn)

        for step, data in enumerate(dataset_real):

            real_images = data['simg'].numpy()

            for i in range(real_images.shape[0]):
                for j in range(real_images.shape[1]):
                    cv2.imwrite(os.path.join(self.real_base, str(step*batch_size + i)+'_'+str(j)+'.png'), 255*(real_images[i, j]))

        return self.real_base, self.fake_base

    @torch.inference_mode()
    def _generate_page(self, ST, SLEN, style_text, eval_text_encode=None, eval_len_text=None):
        SLEN = SLEN.cpu().numpy()
        if eval_text_encode == None:
            eval_text_encode = self.eval_text_encode
        if eval_len_text == None:
            eval_len_text = self.eval_len_text

        style_labels = [' '.join(b.decode() for b in blist).encode() for blist in style_text]
        style_text_encode, style_len_text = self.netconverter.encode(style_labels)

        self.fakes = []
        for i in range(eval_text_encode.size(1)):
            fakes = self.netG(ST, style_text_encode, eval_text_encode[:, i, :])
            self.fakes.append(fakes)

        page1s = []
        page2s = []

        for batch_idx in range(batch_size):

            word_t = []
            word_l = []

            gap = np.ones([IMG_HEIGHT, 16])

            line_wids = []

            for idx, fake_ in enumerate(self.fakes):

                word_t.append((fake_[batch_idx, 0, :, :eval_len_text[idx]*resolution].cpu().numpy()+1)/2)

                word_t.append(gap)

                if len(word_t) == 16 or idx == len(self.fakes) - 1:

                    line_ = np.concatenate(word_t, -1)

                    word_l.append(line_)
                    line_wids.append(line_.shape[1])

                    word_t = []

            gap_h = np.ones([16, max(line_wids)])

            page_ = []

            for l in word_l:

                pad_ = np.ones([IMG_HEIGHT, max(line_wids) - l.shape[1]])

                page_.append(np.concatenate([l, pad_], 1))
                page_.append(gap_h)

            page1 = np.concatenate(page_, 0)

            word_t = []
            word_l = []

            gap = np.ones([IMG_HEIGHT, 16])

            line_wids = []

            for idx in range(SLEN.shape[1]):
                start = int(SLEN[batch_idx, :idx].sum())
                width = int(SLEN[batch_idx, idx])
                word_t.append((ST[batch_idx, :, start:start+width].cpu().numpy()+1)/2)

                word_t.append(gap)

                if len(word_t) == 16 or idx == SLEN.shape[1] - 1:

                    line_ = np.concatenate(word_t, -1)

                    word_l.append(line_)
                    line_wids.append(line_.shape[1])

                    word_t = []

            gap_h = np.ones([16, max(line_wids)])

            page_ = []

            for l in word_l:

                pad_ = np.ones([IMG_HEIGHT, max(line_wids) - l.shape[1]])

                page_.append(np.concatenate([l, pad_], 1))
                page_.append(gap_h)

            page2 = np.concatenate(page_, 0)

            merge_w_size = max(page1.shape[0], page2.shape[0])

            if page1.shape[0] != merge_w_size:

                page1 = np.concatenate([page1, np.ones([merge_w_size-page1.shape[0], page1.shape[1]])], 0)

            if page2.shape[0] != merge_w_size:

                page2 = np.concatenate([page2, np.ones([merge_w_size-page2.shape[0], page2.shape[1]])], 0)

            page1s.append(page1)
            page2s.append(page2)

            # page = np.concatenate([page2, page1], 1)

        page1s_ = np.concatenate(page1s, 0)
        max_wid = max([i.shape[1] for i in page2s])
        padded_page2s = []

        for para in page2s:
            padded_page2s.append(np.concatenate([para, np.ones([para.shape[0], max_wid-para.shape[1]])], 1))

        padded_page2s_ = np.concatenate(padded_page2s, 0)

        return np.concatenate([padded_page2s_, page1s_], 1)

    def get_current_losses(self):

        losses = {}

        losses['G'] = self.loss_G
        losses['D'] = self.loss_D
        losses['Dfake'] = self.loss_Dfake
        losses['Dreal'] = self.loss_Dreal
        losses['OCR_fake'] = self.loss_OCR_fake
        losses['OCR_real'] = self.loss_OCR_real
        losses['w_fake'] = self.loss_w_fake
        losses['w_real'] = self.loss_w_real
        losses['cycle1'] = self.Lcycle1
        losses['cycle2'] = self.Lcycle2
        losses['lda1'] = self.lda1
        losses['lda2'] = self.lda2
        losses['KLD'] = self.KLD

        return losses

    def load_networks(self, epoch):
        BaseModel.load_networks(self, epoch)
        if self.opt.single_writer:
            load_filename = '%s_z.pkl' % (epoch)
            load_path = os.path.join(self.save_dir, load_filename)
            self.z = torch.load(load_path)

    def _set_input(self, input):
        self.input = input

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self):
        self.real = self.input['img'].to(DEVICE)
        self.label = self.input['label']
        self.sdata = self.input['simg'].to(DEVICE)
        self.ST_LEN = self.input['swids']
        self.text_encode, self.len_text = self.netconverter.encode(self.label)
        self.one_hot_real = make_one_hot(self.text_encode, self.len_text, VOCAB_SIZE).to(DEVICE).detach()
        self.text_encode = self.text_encode.to(DEVICE).detach()
        self.len_text = self.len_text.detach()

        self.words = [word.encode('utf-8') for word in random.choices(self.lex, k=batch_size)]
        self.text_encode_fake, self.len_text_fake = self.netconverter.encode(self.words)
        self.text_encode_fake = self.text_encode_fake.to(DEVICE)
        self.one_hot_fake = make_one_hot(self.text_encode_fake, self.len_text_fake, VOCAB_SIZE).to(DEVICE)

        # self.text_encode_fake_js = []

        # for _ in range(NUM_WORDS - 1):
        #     self.words_j = [word.encode('utf-8') for word in random.choices(self.lex, k=batch_size)]
        #     self.text_encode_fake_j, self.len_text_fake_j = self.netconverter.encode(self.words_j)
        #     self.text_encode_fake_j = self.text_encode_fake_j.to(DEVICE)
        #     self.text_encode_fake_js.append(self.text_encode_fake_j)

        self.style_labels = [' '.join(b.decode() for b in blist).encode() for blist in self.input['slbl']]
        self.style_text_encode, self.style_len_text = self.netconverter.encode(self.style_labels)

        self.fake = self.netG(self.sdata, self.style_text_encode, self.text_encode_fake)

    def backward_D_OCR(self):

        pred_real = self.netD(self.real.detach())

        pred_fake = self.netD(**{'x': self.fake.detach()})

        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(), self.len_text.detach(), True)

        self.loss_D = self.loss_Dreal + self.loss_Dfake

        self.pred_real_OCR = self.netOCR(self.real.detach())
        preds_size = torch.IntTensor([self.pred_real_OCR.size(0)] * batch_size).detach()
        loss_OCR_real = self.OCR_criterion(self.pred_real_OCR, self.text_encode.detach(), preds_size, self.len_text.detach())
        self.loss_OCR_real = torch.mean(loss_OCR_real[~torch.isnan(loss_OCR_real)])

        loss_total = self.loss_D + self.loss_OCR_real
        # backward
        loss_total.backward()
        for param in self.netOCR.parameters():
            param.grad[param.grad != param.grad] = 0
            param.grad[torch.isnan(param.grad)] = 0
            param.grad[torch.isinf(param.grad)] = 0

        return loss_total

    def backward_D_WL(self):
        # Real
        pred_real = self.netD(self.real.detach())

        pred_fake = self.netD(**{'x': self.fake.detach()})

        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(), self.len_text.detach(), True)

        self.loss_D = self.loss_Dreal + self.loss_Dfake

        self.loss_w_real = self.netW(self.real.detach(), self.input['wcl'].to(DEVICE)).mean()
        # total loss
        loss_total = self.loss_D + self.loss_w_real

        # backward
        loss_total.backward()

        return loss_total

    def optimize_D_WL(self):
        self.forward()
        self.set_requires_grad([self.netD], True)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], True)

        self.optimizer_D.zero_grad()
        self.optimizer_wl.zero_grad()

        self.backward_D_WL()

    def backward_D_OCR_WL(self):
        # Real
        if self.real_z_mean is None:
            pred_real = self.netD(self.real.detach())
        else:
            pred_real = self.netD(**{'x': self.real.detach(), 'z': self.real_z_mean.detach()})
        # Fake
        try:
            pred_fake = self.netD(**{'x': self.fake.detach(), 'z': self.z.detach()})
        except:
            print('a')
        # Combined loss
        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(), self.len_text.detach(), self.opt.mask_loss)

        self.loss_D = self.loss_Dreal + self.loss_Dfake
        # OCR loss on real data
        self.pred_real_OCR = self.netOCR(self.real.detach())
        preds_size = torch.IntTensor([self.pred_real_OCR.size(0)] * self.opt.batch_size).detach()
        loss_OCR_real = self.OCR_criterion(self.pred_real_OCR, self.text_encode.detach(), preds_size, self.len_text.detach())
        self.loss_OCR_real = torch.mean(loss_OCR_real[~torch.isnan(loss_OCR_real)])
        # total loss
        self.loss_w_real = self.netW(self.real.detach(), self.wcl)
        loss_total = self.loss_D + self.loss_OCR_real + self.loss_w_real

        # backward
        loss_total.backward()
        for param in self.netOCR.parameters():
            param.grad[param.grad != param.grad] = 0
            param.grad[torch.isnan(param.grad)] = 0
            param.grad[torch.isinf(param.grad)] = 0

        return loss_total

    def optimize_D_WL_step(self):
        self.optimizer_D.step()
        self.optimizer_wl.step()
        self.optimizer_D.zero_grad()
        self.optimizer_wl.zero_grad()

    def backward_OCR(self):
        # OCR loss on real data
        self.pred_real_OCR = self.netOCR(self.real.detach())
        preds_size = torch.IntTensor([self.pred_real_OCR.size(0)] * self.opt.batch_size).detach()
        loss_OCR_real = self.OCR_criterion(self.pred_real_OCR, self.text_encode.detach(), preds_size, self.len_text.detach())
        self.loss_OCR_real = torch.mean(loss_OCR_real[~torch.isnan(loss_OCR_real)])

        # backward
        self.loss_OCR_real.backward()
        for param in self.netOCR.parameters():
            param.grad[param.grad != param.grad] = 0
            param.grad[torch.isnan(param.grad)] = 0
            param.grad[torch.isinf(param.grad)] = 0

        return self.loss_OCR_real

    def backward_D(self):
        # Real
        if self.real_z_mean is None:
            pred_real = self.netD(self.real.detach())
        else:
            pred_real = self.netD(**{'x': self.real.detach(), 'z': self.real_z_mean.detach()})
        pred_fake = self.netD(**{'x': self.fake.detach(), 'z': self.z.detach()})
        # Combined loss
        self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(), self.len_text.detach(), self.opt.mask_loss)
        self.loss_D = self.loss_Dreal + self.loss_Dfake
        # backward
        self.loss_D.backward()

        return self.loss_D

    def backward_G_only(self):

        self.gb_alpha = 0.7
        # self.Lcycle1 = self.Lcycle1.mean()
        # self.Lcycle2 = self.Lcycle2.mean()
        self.loss_G = loss_hinge_gen(self.netD(**{'x': self.fake}), self.len_text_fake.detach(), True).mean()

        pred_fake_OCR = self.netOCR(self.fake)
        preds_size = torch.IntTensor([pred_fake_OCR.size(0)] * batch_size).detach()
        loss_OCR_fake = self.OCR_criterion(pred_fake_OCR, self.text_encode_fake.detach(), preds_size, self.len_text_fake.detach())
        self.loss_OCR_fake = torch.mean(loss_OCR_fake[~torch.isnan(loss_OCR_fake)])

        self.loss_G = self.loss_G + self.Lcycle1 + self.Lcycle2 + self.lda1 + self.lda2 - self.KLD

        self.loss_T = self.loss_G + self.loss_OCR_fake

        grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, retain_graph=True)[0]

        self.loss_grad_fake_OCR = 10**6*torch.mean(grad_fake_OCR**2)
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, retain_graph=True)[0]
        self.loss_grad_fake_adv = 10**6*torch.mean(grad_fake_adv**2)

        self.loss_T.backward(retain_graph=True)

        grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=True, retain_graph=True)[0]
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=True, retain_graph=True)[0]

        a = self.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon+torch.std(grad_fake_OCR))

        if a is None:
            print(self.loss_OCR_fake, self.loss_G, torch.std(grad_fake_adv), torch.std(grad_fake_OCR))
        if a > 1000 or a < 0.0001:
            print(a)

        self.loss_OCR_fake = a.detach() * self.loss_OCR_fake

        self.loss_T = self.loss_G + self.loss_OCR_fake

        self.loss_T.backward(retain_graph=True)
        grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=False, retain_graph=True)[0]
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=False, retain_graph=True)[0]
        self.loss_grad_fake_OCR = 10 ** 6 * torch.mean(grad_fake_OCR ** 2)
        self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)

        with torch.no_grad():
            self.loss_T.backward()

        if any(torch.isnan(loss_OCR_fake)) or torch.isnan(self.loss_G):
            print('loss OCR fake: ', loss_OCR_fake, ' loss_G: ', self.loss_G, ' words: ', self.words)
            sys.exit()

    def backward_G_WL(self):

        self.gb_alpha = 0.7
        # self.Lcycle1 = self.Lcycle1.mean()
        # self.Lcycle2 = self.Lcycle2.mean()

        self.loss_G = loss_hinge_gen(self.netD(**{'x': self.fake}), self.len_text_fake.detach(), True).mean()

        self.loss_w_fake = self.netW(self.fake, self.input['wcl'].to(DEVICE)).mean()

        self.loss_G = self.loss_G + self.Lcycle1 + self.Lcycle2 + self.lda1 + self.lda2 - self.KLD

        self.loss_T = self.loss_G + self.loss_w_fake

        self.loss_T.backward(retain_graph=True)

        grad_fake_WL = torch.autograd.grad(self.loss_w_fake, self.fake, create_graph=True, retain_graph=True)[0]
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=True, retain_graph=True)[0]

        a = self.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon+torch.std(grad_fake_WL))

        if a is None:
            print(self.loss_w_fake, self.loss_G, torch.std(grad_fake_adv), torch.std(grad_fake_WL))
        if a > 1000 or a < 0.0001:
            print(a)

        self.loss_w_fake = a.detach() * self.loss_w_fake

        self.loss_T = self.loss_G + self.loss_w_fake

        self.loss_T.backward(retain_graph=True)
        grad_fake_WL = torch.autograd.grad(self.loss_w_fake, self.fake, create_graph=False, retain_graph=True)[0]
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.fake, create_graph=False, retain_graph=True)[0]
        self.loss_grad_fake_WL = 10 ** 6 * torch.mean(grad_fake_WL ** 2)
        self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)

        with torch.no_grad():
            self.loss_T.backward()

    def backward_G(self):
        self.opt.gb_alpha = 0.7
        self.loss_G = loss_hinge_gen(self.netD(**{'x': self.fake, 'z': self.z}), self.len_text_fake.detach(), self.opt.mask_loss)
        # OCR loss on real data

        pred_fake_OCR = self.netOCR(self.fake)
        preds_size = torch.IntTensor([pred_fake_OCR.size(0)] * self.opt.batch_size).detach()
        loss_OCR_fake = self.OCR_criterion(pred_fake_OCR, self.text_encode_fake.detach(), preds_size, self.len_text_fake.detach())
        self.loss_OCR_fake = torch.mean(loss_OCR_fake[~torch.isnan(loss_OCR_fake)])

        self.loss_w_fake = self.netW(self.fake, self.wcl)
        # self.loss_OCR_fake = self.loss_OCR_fake + self.loss_w_fake
        # total loss

       # l1 = self.params[0]*self.loss_G
       # l2 = self.params[0]*self.loss_OCR_fake
        # l3 = self.params[0]*self.loss_w_fake
        self.loss_G_ = 10*self.loss_G + self.loss_w_fake
        self.loss_T = self.loss_G_ + self.loss_OCR_fake

        grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, retain_graph=True)[0]

        self.loss_grad_fake_OCR = 10**6*torch.mean(grad_fake_OCR**2)
        grad_fake_adv = torch.autograd.grad(self.loss_G_, self.fake, retain_graph=True)[0]
        self.loss_grad_fake_adv = 10**6*torch.mean(grad_fake_adv**2)

        if not False:

            self.loss_T.backward(retain_graph=True)

            grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=True, retain_graph=True)[0]
            grad_fake_adv = torch.autograd.grad(self.loss_G_, self.fake, create_graph=True, retain_graph=True)[0]
            # grad_fake_wl = torch.autograd.grad(self.loss_w_fake, self.fake, create_graph=True, retain_graph=True)[0]

            a = self.opt.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon+torch.std(grad_fake_OCR))

            # a0 = self.opt.gb_alpha * torch.div(torch.std(grad_fake_adv), self.epsilon+torch.std(grad_fake_wl))

            if a is None:
                print(self.loss_OCR_fake, self.loss_G_, torch.std(grad_fake_adv), torch.std(grad_fake_OCR))
            if a > 1000 or a < 0.0001:
                print(a)
            b = self.opt.gb_alpha * (torch.mean(grad_fake_adv) -
                                     torch.div(torch.std(grad_fake_adv), self.epsilon+torch.std(grad_fake_OCR)) *
                                     torch.mean(grad_fake_OCR))
            # self.loss_OCR_fake = a.detach() * self.loss_OCR_fake + b.detach() * torch.sum(self.fake)
            self.loss_OCR_fake = a.detach() * self.loss_OCR_fake
            # self.loss_w_fake = a0.detach() * self.loss_w_fake

            self.loss_T = (1-1*self.opt.onlyOCR)*self.loss_G_ + self.loss_OCR_fake  # + self.loss_w_fake
            self.loss_T.backward(retain_graph=True)
            grad_fake_OCR = torch.autograd.grad(self.loss_OCR_fake, self.fake, create_graph=False, retain_graph=True)[0]
            grad_fake_adv = torch.autograd.grad(self.loss_G_, self.fake, create_graph=False, retain_graph=True)[0]
            self.loss_grad_fake_OCR = 10 ** 6 * torch.mean(grad_fake_OCR ** 2)
            self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)
            with torch.no_grad():
                self.loss_T.backward()
        else:
            self.loss_T.backward()

        if self.opt.clip_grad > 0:
            clip_grad_norm_(self.netG.parameters(), self.opt.clip_grad)
        if any(torch.isnan(loss_OCR_fake)) or torch.isnan(self.loss_G_):
            print('loss OCR fake: ', loss_OCR_fake, ' loss_G: ', self.loss_G, ' words: ', self.words)
            sys.exit()

    def optimize_D_OCR(self):
        self.forward()
        self.set_requires_grad([self.netD], True)
        self.set_requires_grad([self.netOCR], True)
        self.optimizer_D.zero_grad()
        # if self.opt.OCR_init in ['glorot', 'xavier', 'ortho', 'N02']:
        self.optimizer_OCR.zero_grad()
        self.backward_D_OCR()

    def optimize_OCR(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netOCR], True)
        if self.opt.OCR_init in ['glorot', 'xavier', 'ortho', 'N02']:
            self.optimizer_OCR.zero_grad()
        self.backward_OCR()

    def optimize_D(self):
        self.forward()
        self.set_requires_grad([self.netD], True)
        self.backward_D()

    def optimize_D_OCR_step(self):
        self.optimizer_D.step()

        self.optimizer_OCR.step()
        self.optimizer_D.zero_grad()
        self.optimizer_OCR.zero_grad()

    def optimize_D_OCR_WL(self):
        self.forward()
        self.set_requires_grad([self.netD], True)
        self.set_requires_grad([self.netOCR], True)
        self.set_requires_grad([self.netW], True)
        self.optimizer_D.zero_grad()
        self.optimizer_wl.zero_grad()
        if self.opt.OCR_init in ['glorot', 'xavier', 'ortho', 'N02']:
            self.optimizer_OCR.zero_grad()
        self.backward_D_OCR_WL()

    def optimize_D_OCR_WL_step(self):
        self.optimizer_D.step()
        if self.opt.OCR_init in ['glorot', 'xavier', 'ortho', 'N02']:
            self.optimizer_OCR.step()
        self.optimizer_wl.step()
        self.optimizer_D.zero_grad()
        self.optimizer_OCR.zero_grad()
        self.optimizer_wl.zero_grad()

    def optimize_D_step(self):
        self.optimizer_D.step()
        if any(torch.isnan(self.netD.infer_img.blocks[0][0].conv1.bias)):
            print('D is nan')
            sys.exit()
        self.optimizer_D.zero_grad()

    def optimize_G(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], False)
        self.backward_G()

    def optimize_G_WL(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], False)
        self.backward_G_WL()

    def optimize_G_only(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netOCR], False)
        self.set_requires_grad([self.netW], False)
        self.backward_G_only()

    def optimize_G_step(self):

        self.optimizer_G.step()
        self.optimizer_G.zero_grad()

    def optimize_ocr(self):
        self.set_requires_grad([self.netOCR], True)
        # OCR loss on real data
        pred_real_OCR = self.netOCR(self.real)
        preds_size = torch.IntTensor([pred_real_OCR.size(0)] * self.opt.batch_size).detach()
        self.loss_OCR_real = self.OCR_criterion(pred_real_OCR, self.text_encode.detach(), preds_size, self.len_text.detach())
        self.loss_OCR_real.backward()
        self.optimizer_OCR.step()

    def optimize_z(self):
        self.set_requires_grad([self.z], True)

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def test(self):
        self.visual_names = ['fake']
        self.netG.eval()
        with torch.no_grad():
            self.forward()

    def train_GD(self):
        self.netG.train()
        self.netD.train()
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()
        # How many chunks to split x and y into?
        x = torch.split(self.real, self.opt.batch_size)
        y = torch.split(self.label, self.opt.batch_size)
        counter = 0

        # Optionally toggle D and G's "require_grad"
        if self.opt.toggle_grads:
            toggle_grad(self.netD, True)
            toggle_grad(self.netG, False)

        for step_index in range(self.opt.num_critic_train):
            self.optimizer_D.zero_grad()
            with torch.set_grad_enabled(False):
                self.forward()
            D_input = torch.cat([self.fake, x[counter]], 0) if x is not None else self.fake
            D_class = torch.cat([self.label_fake, y[counter]], 0) if y[counter] is not None else y[counter]
            # Get Discriminator output
            D_out = self.netD(D_input, D_class)
            if x is not None:
                pred_fake, pred_real = torch.split(D_out, [self.fake.shape[0], x[counter].shape[0]])  # D_fake, D_real
            else:
                pred_fake = D_out
            # Combined loss
            self.loss_Dreal, self.loss_Dfake = loss_hinge_dis(pred_fake, pred_real, self.len_text_fake.detach(), self.len_text.detach(), self.opt.mask_loss)
            self.loss_D = self.loss_Dreal + self.loss_Dfake
            self.loss_D.backward()
            counter += 1
            self.optimizer_D.step()

        # Optionally toggle D and G's "require_grad"
        if self.opt.toggle_grads:
            toggle_grad(self.netD, False)
            toggle_grad(self.netG, True)
        # Zero G's gradients by default before training G, for safety
        self.optimizer_G.zero_grad()
        self.forward()
        self.loss_G = loss_hinge_gen(self.netD(self.fake, self.label_fake), self.len_text_fake.detach(), self.opt.mask_loss)
        self.loss_G.backward()
        self.optimizer_G.step()

    def save_networks(self, epoch, save_dir):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    # torch.save(net.module.cpu().state_dict(), save_path)
                    if len(self.gpu_ids) > 1:
                        torch.save(net.module.cpu().state_dict(), save_path)
                    else:
                        torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
