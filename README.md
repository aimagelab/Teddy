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
(teddy) vpippi@collepino:~/Teddy$ python train.py --epochs_size 200 --dryrun
Loading datasets train: 100%|█████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.93s/it]
Dataset has 25592 samples.
initialize network with N02
Teddy has 140.41 M parameters.
Epoch 0: 100%|████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:43<00:00,  4.58it/s]
Epoch 1: 100%|████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:41<00:00,  4.82it/s]
Epoch 2: 100%|████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:42<00:00,  4.67it/s]
Epoch 3: 100%|████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:44<00:00,  4.54it/s]
Epoch 4: 100%|████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:44<00:00,  4.46it/s]
Epoch 5: 100%|████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:45<00:00,  4.38it/s]
Epoch 6: 100%|████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:46<00:00,  4.30it/s]
Epoch 7: 100%|████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:46<00:00,  4.27it/s]
Epoch 8: 100%|████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:46<00:00,  4.28it/s]
Epoch 9: 100%|████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:46<00:00,  4.26it/s]
Epoch 10: 100%|███████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:47<00:00,  4.21it/s]
...
```
