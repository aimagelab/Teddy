import torch
import argparse
import random
import numpy as np
from model.teddy import Teddy


def train(args):
    dataset = None
    loader = None

    teddy = Teddy(dataset.vocab_size, dim=512).to(args.device)
    optimizer = torch.optim.AdamW(teddy.parameters(), lr=args.lr)

    # TODO losses

    for epoch in range(args.epochs):
        pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=742)
    args = parser.parse_args()

    set_seed(args.seed)
    train(args)

    # import string

    # device = 'cuda'
    # teddy = Teddy(string.ascii_lowercase + ' ' + string.digits, dim=512).to(device)
    # # compute parameters
    # params = sum(p.numel() for p in teddy.parameters() if p.requires_grad)
    # print(f'The model has {params} trainable parameters')

    # style_imgs = torch.cat([torch.ones((args.batch_size, 3, 32, 16)) * i for i in range(57)], dim=-1).to(device)
    # style_imgs_len = torch.randint(30, 57, (args.batch_size, )).to(device) * 16
    # same_author_imgs = torch.cat([torch.ones((args.batch_size, 3, 32, 16)) * i for i in range(100)], dim=-1).to(device)
    # same_author_imgs_len = torch.randint(30, 100, (args.batch_size, )).to(device) * 16
    # other_author_imgs = torch.cat([torch.ones((args.batch_size, 3, 32, 16)) * i for i in range(67)], dim=-1).to(device)
    # other_author_imgs_len = torch.randint(30, 67, (args.batch_size, )).to(device) * 16
    # style_text = [f'style ciao{i**i}' for i in range(args.batch_size)]
    # gen_text = [f'ciao{i**i}' for i in range(args.batch_size)]
    # out = teddy(style_imgs, style_imgs_len, style_text, gen_text, same_author_imgs, same_author_imgs_len, other_author_imgs, other_author_imgs_len)
    # print('No errors')
