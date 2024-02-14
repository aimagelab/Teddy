## Installation
```bash
git clone https://github.com/aimagelab/Teddy.git && cd Teddy
```

```bash
conda create --name teddy python==3.11.5
conda activate teddy
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

Download the files necessary to run the code:
```bash
wget https://github.com/aimagelab/Teddy/releases/download/files/iam_eval.zip
unzip iam_eval.zip
rm iam_eval.zip

wget https://github.com/aimagelab/Teddy/releases/download/files/ocr_checkpoints.zip
unzip ocr_checkpoints.zip
rm ocr_checkpoints.zip
```

Download the dataset:
```bash
cd /folder/to/datasets
wget -i https://github.com/aimagelab/Teddy/releases/download/datasets/urls.txt
zip -F IAM.zip --out IAM-single.zip && unzip IAM-single.zip
rm IAM.zip IAM.z0* IAM-single.zip urls.txt
```

## Training
```bash
python train.py --ddp --world_size 2 --root_path /folder/to/datasets
```
