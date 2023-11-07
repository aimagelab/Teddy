from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

src = Path('/mnt/ssd/datasets/IAM/lines_16')

sizes = [Image.open(img_path).size for img_path in tqdm(list(src.rglob('*.png')))]
height = 32
widths = [int(size[0] * height / size[1]) for size in sizes]

avg = sum(widths) / len(widths)
perc_80 = sorted(widths)[int(len(widths) * 0.8)]

plt.hist(widths, bins=100)
# add thick in the average width
plt.axvline(avg, color='red')
# add thick in 75% of the width
plt.axvline(perc_80, color='green')
plt.xlabel('width')
plt.ylabel('count')

# change thicks on x
# plt.xticks([avg, perc_75], [f'{int(avg):d}', f'{int(perc_75):d}'])
# one thick every 50 on x
plt.xticks(range(0, 1000, 100), [f'{x:d}' for x in range(0, 1000, 100)])
plt.title(f'IAM width distribution\navg: {avg:.0f}, 80%: {perc_80:.0f}, max: {max(widths):.0f}')
plt.savefig('iam_width_distribution.png')
