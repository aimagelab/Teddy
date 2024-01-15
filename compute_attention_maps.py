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


class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def _mha_block(self, x, mem, attn_mask, key_padding_mask, is_causal):
        x, self.attention_map = self.multihead_attn(x, mem, mem,
                                                    attn_mask=attn_mask,
                                                    key_padding_mask=key_padding_mask,
                                                    is_causal=is_causal,
                                                    need_weights=True)
        return self.dropout2(x)


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
        last_checkpoint = sorted(args.checkpoint_path.glob('*_epochs.pth'))[-1]
        checkpoint = torch.load(last_checkpoint)
        teddy.load_state_dict(checkpoint['model'])
    else:
        raise ValueError(f"Checkpoint path {args.checkpoint_path} does not exist")

    for sample_idx, sample in enumerate(tqdm(dataset)):
        try:
            batch = dataset.collate_fn([sample])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch['gen_texts'] = ['little',]

            fakes = teddy.generate(batch['gen_texts'], batch['style_texts'], batch['style_imgs'])

            attention_maps = []
            for layer in teddy.generator.transformer_gen_decoder.layers:
                attention_maps.append(layer.attention_map.detach().cpu())
            attention_maps = torch.cat(attention_maps, dim=1).squeeze(0)
            attention_maps /= attention_maps.max()
            attention_ctrl = torch.Tensor([c in set(batch['gen_texts'][0]) for c in batch['style_texts'][0]])
            attention_maps = torch.cat([attention_ctrl.unsqueeze(0), attention_maps], dim=0)

            dst = Path(args.checkpoint_path, 'attention_maps')
            dst.mkdir(exist_ok=True)

            save_image(attention_maps, Path(dst, f'layer_{sample_idx}.png'))
            with open(Path(dst, f'layer_{sample_idx}.txt'), 'w') as f:
                f.write(batch['gen_texts'][0])
                f.write(f'\n{len(batch["gen_texts"][0])}\n')
                f.write(batch['style_texts'][0])
                f.write(f'\n{len(batch["style_texts"][0])}\n')
        except KeyboardInterrupt as e:
            break


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
