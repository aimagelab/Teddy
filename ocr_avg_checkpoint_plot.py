import torch
from model.ocr import OrigamiNet
from model.teddy import CTCLabelConverter
from datasets import dataset_factory
from util import CheckpointScheduler
from tqdm import tqdm
from matplotlib import pyplot as plt
import editdistance
import numpy as np

device = 'cuda:0'

checkpoint_a = torch.load('files/f745_all_datasets/0345000_iter.pth', map_location=device)
checkpoint_b = torch.load('files/0ea8_all_datasets/0345000_iter.pth', map_location=device)

o_classes = len(checkpoint_a['charset'])

converter = CTCLabelConverter(''.join(sorted(checkpoint_a['charset'])))
ocr = OrigamiNet(o_classes + 1).to(device)
# checkpoint_a['model'] = {k: v.half() for k, v in checkpoint_a['model'].items()}
# checkpoint_b['model'] = {k: v.half() for k, v in checkpoint_b['model'].items()}
ocr_scheduler = CheckpointScheduler(ocr, checkpoint_a['model'], checkpoint_b['model'])
ocr.eval()

dataset_name, dataset_path = ['iam_lines'], ['/mnt/ssd/datasets/IAM']
# dataset_name, dataset_path = ['lam'], ['/mnt/ssd/datasets/LAM_msgpack']
dataset = dataset_factory(dataset_name, dataset_path, 'val', max_width=3000)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)
ctc_criterion = torch.nn.CTCLoss(reduction='sum', zero_infinity=True).to(device)


loss_scores = []
cer_scores = []
alphas = []


with torch.inference_mode():
    steps = 20

    pbar = tqdm(total=steps * len(loader))
    for alpha_idx, alpha in enumerate(torch.linspace(0, 1, steps).tolist()):
        total_loss = 0
        total_count = 0

        n_correct = 0
        norm_ED = 0
        tot_ED = 0
        length_of_gt = 0
        infer_time = 0

        ocr_scheduler._step(alpha)
        for batch_idx, batch in enumerate(loader):
            style_imgs = batch['style_imgs'].to(device)
            texts_enc, texts_enc_len = converter.encode(batch['style_texts'])

            # fakes_exp = rearrange(fakes, 'b e c h w -> (b e) c h w')
            texts_pred = ocr(style_imgs)

            b, w, _ = texts_pred.shape
            preds_size = torch.IntTensor([w] * b).to(device)
            texts_pred = texts_pred.permute(1, 0, 2).log_softmax(2)

            torch.backends.cudnn.enabled = False
            ctc_loss = ctc_criterion(texts_pred, texts_enc, preds_size, texts_enc_len)
            torch.backends.cudnn.enabled = True

            _, preds_index = texts_pred.max(2)
            preds_index = preds_index.transpose(1, 0).cpu().numpy()
            preds_size = preds_size.cpu().numpy() - (np.flip(preds_index, 1) > 0).argmax(-1)
            preds_str = converter.decode(preds_index, preds_size)

            total_loss += ctc_loss.item()
            total_count += len(batch['style_texts'])

            for pred, gt in zip(preds_str, batch['style_texts']):
                tmped = editdistance.eval(pred, gt)
                if pred == gt:
                    n_correct += 1
                if len(gt) == 0:
                    norm_ED += 1
                else:
                    norm_ED += tmped / float(len(gt))

                tot_ED += tmped
                length_of_gt += len(gt)

            pbar.update(1)
        # print(alpha, total_loss / total_count)
        alphas.append(alpha)
        loss_scores.append(total_loss / total_count)

        tot_ED = tot_ED / float(length_of_gt)
        norm_ED /= total_count
        accuracy = n_correct / total_count * 100

        cer_scores.append(norm_ED)


# Create the figure and first axis
fig, ax1 = plt.subplots()

# Plot the first dataset on the primary axis
ax1.plot(alphas, loss_scores, color='b', label='CTC Loss', linewidth=2)
ax1.set_xlabel('Alphas')
ax1.set_ylabel('CTC Loss', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a second axis sharing the same x-axis
ax2 = ax1.twinx()

# Plot the second dataset on the secondary axis
ax2.plot(alphas, cer_scores, color='r', label='cer', linewidth=2)
ax2.set_ylabel('CER', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Customize the appearance of the plot
ax1.set_title('Dual Y-Axis Plot')
ax1.grid(True)

# Add a legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.savefig(f'files/{dataset_name[0]}_ocr_avg_checkpoint_plot.png')
