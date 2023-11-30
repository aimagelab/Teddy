## Installation
```bash
conda create --name teddy python==3.11.5
conda activate teddy
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

```bash
cd Teddy
wget https://github.com/aimagelab/Teddy/releases/download/files/files.zip
unzip files.zip
rm files.zip
```