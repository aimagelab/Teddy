## Installation
```bash
git clone https://github.com/aimagelab/Teddy.git && cd Teddy
```

```bash
conda create --name teddy python==3.11.5
conda activate teddy
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
python -m nltk.downloader all
```

Download the files necessary to run the code:
```bash
wget https://github.com/aimagelab/Teddy/releases/download/files/iam_eval.zip
unzip iam_eval.zip
rm iam_eval.zip

wget https://github.com/aimagelab/Teddy/releases/download/files/ocr_checkpoints.zip
unzip ocr_checkpoints.zip
rm ocr_checkpoints.zip

wget https://github.com/aimagelab/Teddy/releases/download/files/iam_htg_setting.json.gz -P files
```

Download HWD:
```bash
git clone https://github.com/aimagelab/HWD hwd
```

Download the dataset:
```bash
wget https://github.com/aimagelab/Teddy/releases/download/pkl/iam_lines_l_train.pkl -P files/datasets
```

## Training
```bash
python train.py --wandb
```
The `--wandb` flag enables the log on wandb.

## Test the code
```bash
python train.py --dryrun
```
The `--dryrun` flag suppresses the logging on wandb if set and disables the checkpoint saving.

## Time incremental bug
```bash
(teddy) vpippi@collepino:~/Teddy$ python train.py --dryrun
Loading datasets train: 100%|█████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.91s/it]
Dataset has 25592 samples.
initialize network with N02
Teddy has 140.41 M parameters.
Epoch 0: 100%|██████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [03:36<00:00,  4.63it/s]
Epoch 1: 100%|██████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [03:53<00:00,  4.29it/s]
Epoch 2: 100%|██████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [03:57<00:00,  4.21it/s]
...
```
