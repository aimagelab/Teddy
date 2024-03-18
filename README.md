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
python train.py
```
