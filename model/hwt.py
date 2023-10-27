from torch import nn
import torch
from model.detr_transformer import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
from model.cnn_decoder import FCNDecoder
from torchvision import models


class HWTGenerator(nn.Module):
    def __init__(self, vocab_size):
        super(HWTGenerator, self).__init__()

        INP_CHANNEL = 1

        encoder_layer = TransformerEncoderLayer(512, 8, 512, 0.1, "relu", True)
        encoder_norm = nn.LayerNorm(512) if True else None
        self.encoder = TransformerEncoder(encoder_layer, 3, encoder_norm)

        decoder_layer = TransformerDecoderLayer(512, 8, 512, 0.1, "relu", True)
        decoder_norm = nn.LayerNorm(512)
        self.decoder = TransformerDecoder(decoder_layer, 3, decoder_norm,
                                          return_intermediate=True)

        self.Feat_Encoder = nn.Sequential(*([nn.Conv2d(INP_CHANNEL, 64, kernel_size=7, stride=2, padding=3, bias=False)] + list(models.resnet18(pretrained=True).children())[1:-2]))

        self.query_embed = nn.Embedding(vocab_size, 512)

        self.linear_q = nn.Linear(512, 512*8)

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
        B, N, R, C = ST.shape
        FEAT_ST = self.Feat_Encoder(ST.view(B*N, 1, R, C))
        FEAT_ST = FEAT_ST.view(B, 512, 1, -1)

        FEAT_ST_ENC = FEAT_ST.flatten(2).permute(2, 0, 1)

        memory = self.encoder(FEAT_ST_ENC)

        OUT_IMGS = []

        for i in range(QRS.shape[1]):

            QR = QRS[:, i, :]

            QR_EMB = self.query_embed.weight[QR].permute(1, 0, 2)

            tgt = torch.zeros_like(QR_EMB)

            hs = self.decoder(tgt, memory, query_pos=QR_EMB)

            h = hs.transpose(1, 2)[-1]  # torch.cat([hs.transpose(1, 2)[-1], QR_EMB.permute(1,0,2)], -1)

            h = self.linear_q(h)
            h = h.contiguous()

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

        # h = self.linear_q(h)
        # h = h.contiguous()

        # h = h.view(h.size(0), h.shape[1]*2, 4, -1)
        # h = h.permute(0, 3, 2, 1)

        h = self.DEC(h)

        self.dec_attn_weights = dec_attn_weights[-1].detach()
        self.enc_attn_weights = enc_attn_weights[-1].detach()

        for hook in self.hooks:
            hook.remove()

        return h
