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
cd Teddy
wget https://github.com/aimagelab/Teddy/releases/download/files/files.zip
unzip files.zip
rm files.zip
```

Download the dataset:
```bash
cd /folder/to/datasets
wget https://github.com/aimagelab/Teddy/releases/download/datasets/IAM.zip
wget https://github.com/aimagelab/Teddy/releases/download/datasets/IAM.z01
wget https://github.com/aimagelab/Teddy/releases/download/datasets/IAM.z02
wget https://github.com/aimagelab/Teddy/releases/download/datasets/IAM.z03
wget https://github.com/aimagelab/Teddy/releases/download/datasets/IAM.z04
wget https://github.com/aimagelab/Teddy/releases/download/datasets/IAM.z05
wget https://github.com/aimagelab/Teddy/releases/download/datasets/IAM.z06
zip -F IAM.zip --out IAM-single.zip
unzip IAM-single.zip
rm IAM.zip IAM.z0* 
```

## Training
```bash
python train.py --ddp --world_size 2 --root_path /folder/to/datasets
```