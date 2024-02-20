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

cd files
wget https://github.com/aimagelab/Teddy/releases/download/files/iam_htg_setting.json.gz
```

Download HWD:
```bash
git clone https://github.com/aimagelab/HWD hwd
```

Download the dataset:
```bash
cd /folder/to/datasets
wget -qO- https://github.com/aimagelab/Teddy/releases/download/pkl/iam_lines_sm.tar.gz | tar xvz

wget -qO- https://github.com/aimagelab/Teddy/releases/download/pkl/iam_eval.tar.gz | tar xvz
wget -qO- https://github.com/aimagelab/Teddy/releases/download/pkl/iam_lines.tar.gz | tar xvz
wget -qO- https://github.com/aimagelab/Teddy/releases/download/pkl/iam_lines_16.tar.gz | tar xvz
wget -qO- https://github.com/aimagelab/Teddy/releases/download/pkl/iam_lines_xs.tar.gz | tar xvz
wget -qO- https://github.com/aimagelab/Teddy/releases/download/pkl/iam_lines_xxs.tar.gz | tar xvz
wget -qO- https://github.com/aimagelab/Teddy/releases/download/pkl/iam_words.tar.gz | tar xvz

wget -qO- https://github.com/aimagelab/Teddy/releases/download/pkl/icfhr14.tar.gz | tar xvz
wget -qO- https://github.com/aimagelab/Teddy/releases/download/pkl/icfhr16.tar.gz | tar xvz
wget -qO- https://github.com/aimagelab/Teddy/releases/download/pkl/leopardi.tar.gz | tar xvz
wget -qO- https://github.com/aimagelab/Teddy/releases/download/pkl/norhand.tar.gz | tar xvz
wget -qO- https://github.com/aimagelab/Teddy/releases/download/pkl/rimes.tar.gz | tar xvz
wget -qO- https://github.com/aimagelab/Teddy/releases/download/pkl/saintgall.tar.gz | tar xvz
wget -qO- https://github.com/aimagelab/Teddy/releases/download/pkl/washington.tar.gz | tar xvz
```

## Training
```bash
python train.py --ddp --world_size 2 --root_path /folder/to/datasets
```
