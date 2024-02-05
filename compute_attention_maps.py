import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from train import add_arguments, set_seed, free_mem_percent
from datasets import dataset_factory
from model.teddy import Teddy
from torchvision.utils import save_image
from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt
import string


class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def _mha_block(self, x, mem, attn_mask, key_padding_mask, is_causal):
        x, self.attention_map = self.multihead_attn(x, mem, mem,
                                                    attn_mask=attn_mask,
                                                    key_padding_mask=key_padding_mask,
                                                    is_causal=is_causal,
                                                    need_weights=True)
        return self.dropout2(x)


def save_plot_fig(d, filename):
    keys = sorted(k for k in d.keys() if len(k) == 2)
    keys += sorted(k for k in d.keys() if len(k) == 1)
    values = [d[k] for k in keys]
    plt.bar(keys, values)
    # plt.xticks(rotation=90)
    plt.tight_layout()
    # increase graph width
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    dst = Path(args.checkpoint_path, filename)
    dst.parent.mkdir(exist_ok=True)
    plt.savefig(dst)
    plt.close()


def compute_attention(rank, args):
    device = torch.device(rank)

    dataset = dataset_factory('train', **args.__dict__)

    ocr_checkpoint_a = torch.load(args.ocr_checkpoint_a, map_location=device)
    # ocr_checkpoint_b = torch.load(args.ocr_checkpoint_b, map_location=device)
    # ocr_checkpoint_c = torch.load(args.ocr_checkpoint_c, map_location=device)
    # assert ocr_checkpoint_a['charset'] == ocr_checkpoint_b['charset'], "OCR checkpoints must have the same charset"

    teddy = Teddy(ocr_checkpoint_a['charset'], **args.__dict__)
    l = teddy.generator.transformer_gen_decoder.layers[0]
    layer = CustomTransformerDecoderLayer(d_model=l.self_attn.embed_dim, nhead=l.self_attn.num_heads,
                                          dim_feedforward=l.linear1.out_features, dropout=l.dropout.p, batch_first=l.self_attn.batch_first)
    teddy.generator.transformer_gen_decoder = nn.TransformerDecoder(
        layer, num_layers=teddy.generator.transformer_gen_decoder.num_layers, norm=teddy.generator.transformer_gen_decoder.norm)
    teddy.to(device)

    if args.checkpoint_path.exists() and len(list(args.checkpoint_path.glob('*_epochs.pth'))) > 0:
        # last_checkpoint = sorted(args.checkpoint_path.glob('*_epochs.pth'))[-1]
        # if args.start_epochs > 0 and Path(args.checkpoint_path, f'{args.start_epochs:06d}_epochs.pth').exists():
        #     last_checkpoint = Path(args.checkpoint_path, f'{args.start_epochs:06d}_epochs.pth')
        # checkpoint = torch.load(last_checkpoint)
        # teddy.load_state_dict(checkpoint['model'])
        # args.start_epochs = last_checkpoint.name.split('_')[0]
        # print(f"Loaded checkpoint {last_checkpoint}")
        pass
    else:
        raise ValueError(f"Checkpoint path {args.checkpoint_path} does not exist")

    for checkpoint_path in sorted(args.checkpoint_path.glob('*_epochs.pth'), reverse=True):
        checkpoint = torch.load(checkpoint_path)
        teddy.load_state_dict(checkpoint['model'])
        args.start_epochs = checkpoint_path.name.split('_')[0]
        print(f"Loaded checkpoint {checkpoint_path}")

        teddy.eval()

        for c in string.ascii_lowercase:
            pos_weights_dict = defaultdict(list)
            char_weights_dict = defaultdict(list)
            gen_texts = [c]
            for sample_idx in tqdm(range(1000)):
                sample = dataset[sample_idx]
                try:
                    batch = dataset.collate_fn([sample])
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    batch['gen_text'] = gen_texts

                    fakes = teddy.generate(batch['gen_text'], batch['style_text'], batch['style_img'])

                    attention_maps = []
                    for layer in teddy.generator.transformer_gen_decoder.layers:
                        attention_maps.append(layer.attention_map.detach().cpu())
                    attention_maps = torch.cat(attention_maps, dim=1).squeeze(0)

                    glob_style_tokens = [f'g{i}' for i in range(attention_maps.shape[1] - len(batch['style_text'][0]))]
                    characters = glob_style_tokens + list(batch['style_text'][0])
                    attention_maps = attention_maps.mean(dim=0)
                    for i, c in enumerate(characters):
                        char_weights_dict[c].append(attention_maps[i].item())
                        pos_weights_dict[f'{i:02d}'].append(attention_maps[i].item())

                    # attention_ctrl = torch.Tensor(glob_style_tokens + [c in set(batch['gen_texts'][0]) for c in batch['style_texts'][0]])
                    # attention_maps = torch.cat([attention_ctrl.unsqueeze(0), attention_maps], dim=0)

                    # dst = Path(args.checkpoint_path, 'attention_maps')
                    # dst.mkdir(exist_ok=True)

                    # save_image(attention_maps, Path(dst, f'layer_{sample_idx}.png'))
                    # with open(Path(dst, f'layer_{sample_idx}.txt'), 'w') as f:
                    #     f.write(batch['gen_texts'][0])
                    #     f.write(f'\n{len(batch["gen_texts"][0])}\n')
                    #     f.write(batch['style_texts'][0])
                    #     f.write(f'\n{len(batch["style_texts"][0])}\n')
                except KeyboardInterrupt as e:
                    break

            pos_weights_dict = {k: np.mean(v) for k, v in pos_weights_dict.items()}
            char_weights_dict = {k: np.mean(v) for k, v in char_weights_dict.items()}

            save_plot_fig(pos_weights_dict, f'attention_maps/{args.start_epochs}_pos_weights_{gen_texts[0]}.png')
            save_plot_fig(char_weights_dict, f'attention_maps/{args.start_epochs}_char_weights_{gen_texts[0]}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    args.datasets_path = [Path(args.root_path, path) for path in args.datasets_path]
    args.checkpoint_path = Path(args.checkpoint_path, args.run_id)

    set_seed(args.seed)

    assert torch.cuda.is_available(), "You need a GPU to train Teddy"
    if args.device == 'auto':
        args.device = f'cuda:{np.argmax(free_mem_percent())}'

    compute_attention(args.device, args)
