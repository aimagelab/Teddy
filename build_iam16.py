from pathlib import Path
import argparse
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import numpy as np
import html

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=Path, default='/mnt/ssd/datasets/IAM')
parser.add_argument('--height', type=int, default=32)
parser.add_argument('--width_per_char', type=int, default=16)
args = parser.parse_args()

lines_path = args.path / 'lines'
lines_16_path = args.path / 'lines_16'

lines_id_to_path = {line_path.stem: line_path for line_path in lines_path.rglob('*.png')}
xml_files = [ET.parse(xml_file) for xml_file in (args.path / 'xmls').rglob('*.xml')]
forms = {form_path.stem: form_path for form_path in (args.path / 'forms').rglob('*.png')}

data = {}
for xml_file in tqdm(xml_files):
    root = xml_file.getroot()
    img = Image.open(forms[root.attrib['id']])
    img = np.array(img)
    hpart = root.find('handwritten-part')
    for line_el in hpart.iterfind('line'):
        text = html.unescape(line_el.attrib['text'])
        img_id = line_el.attrib['id']

        src_path = lines_id_to_path[img_id]
        dst_path = lines_16_path / src_path.relative_to(lines_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(src_path)
        img = img.resize((args.width_per_char * len(text), args.height), Image.BILINEAR)
        img.save(dst_path)
