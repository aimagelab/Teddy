from pathlib import Path
import argparse
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import random
from tqdm import tqdm
import numpy as np
import html
import json
import editdistance


def random_color():
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    return "#{:02X}{:02X}{:02X}".format(red, green, blue)


class Coords:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def size(self):
        return self.height, self.width

    @property
    def shape(self):
        return self.x1, self.y1, self.x2, self.y2

    def width_scaled(self, height):
        return int(self.width * height / self.height)

    def __add__(self, other):
        return Coords(
            min(self.x1, other.x1),
            min(self.y1, other.y1),
            max(self.x2, other.x2),
            max(self.y2, other.y2)
        )


def cmp_coords(cmp_el):
    x = int(cmp_el.attrib['x'])
    y = int(cmp_el.attrib['y'])
    w = int(cmp_el.attrib['width'])
    h = int(cmp_el.attrib['height'])
    return Coords(x, y, x + w, y + h)


def word_coords(word_el):
    coords = [cmp_coords(cmp_el) for cmp_el in word_el.iterfind('cmp')]
    if len(coords) == 0:
        return None
    return sum(coords, coords[0])


def indices(text, sub):
    start = 0
    res = []
    while True:
        start = text.find(sub, start)
        if start == -1:
            return res
        res.append(start)
        start += len(sub)


def string_insert(text, idx, sub):
    return text[:idx] + sub + text[idx:]


def search(text, sub):
    for i in range(1, len(sub)):
        word = string_insert(sub, i, ' ')
        if word in text:
            return indices(text, word), word
    return [], sub


def join_words(words, ref):
    start_indices = indices(ref, words[0])
    if len(start_indices) == 0:
        start_indices, word = search(ref, words[0])
        assert len(start_indices) > 0
        words[0] = word
    end_indices = indices(ref, words[-1])
    if len(end_indices) == 0:
        end_indices, word = search(ref, words[-1])
        assert len(end_indices) > 0
        words[-1] = word

    best_text = None
    best_score = 10**10
    no_space_words = ''.join(words).replace(' ', '')

    for start_idx in start_indices:
        for end_idx in end_indices:
            text = ref[start_idx:end_idx + len(words[-1])]
            no_space_text = text.replace(' ', '')
            if no_space_text == no_space_words:
                return text, 0
            dist = editdistance.eval(no_space_text, no_space_words)
            if dist < best_score:
                best_score = dist
                best_text = text
    if best_text is not None:
        return best_text, best_score
    raise ValueError('Cannot find text')


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=Path, default='/mnt/ssd/datasets/IAM')
parser.add_argument('--output', type=str, default='/mnt/ssd/datasets/IAM/lines_sm')
parser.add_argument('--height', type=int, default=32)
parser.add_argument('--min_width', type=int, default=100)
parser.add_argument('--max_width', type=int, default=200)
args = parser.parse_args()


xml_files = [ET.parse(xml_file) for xml_file in (args.path / 'xmls').rglob('*.xml')]
forms = {form_path.stem: form_path for form_path in (args.path / 'forms').rglob('*.png')}

data = {}
for xml_file in tqdm(xml_files):
    root = xml_file.getroot()
    img = Image.open(forms[root.attrib['id']])
    img = np.array(img)
    hpart = root.find('handwritten-part')
    for line_el in hpart.iterfind('line'):
        words = line_el.findall('word')
        if len(words) == 0:
            continue
        words_coords = [word_coords(word) for word in words]
        words, words_coords = zip(*[(word, coords) for word, coords in zip(words, words_coords) if coords is not None])

        line_text = line_el.attrib['text']
        line_text = html.unescape(line_text)
        for start_idx in range(len(words)):
            end_idx = start_idx
            while end_idx < len(words) and sum(words_coords[start_idx:end_idx], words_coords[start_idx]).width_scaled(args.height) < args.min_width:
                end_idx += 1
            color = random_color()
            coords = sum(words_coords[start_idx:end_idx], words_coords[start_idx])
            if args.min_width < coords.width_scaled(args.height) < args.max_width and start_idx < end_idx:
                img_crop = img[coords.y1:coords.y2, coords.x1:coords.x2]
                img_crop = Image.fromarray(img_crop)
                dst_path = Path(args.output, root.attrib['id'], line_el.attrib["id"], f'{line_el.attrib["id"]}-{start_idx:02d}.png')

                try:
                    text, dist = join_words([word_el.attrib['text'] for word_el in words[start_idx:end_idx]], line_text)
                    if dist > 2:
                        print(f'Warning: {line_el.attrib["id"]}-{start_idx:02d} dist={dist} text={text}')
                    data[dst_path.stem] = {
                        'text': text,
                        'auth': root.attrib['writer-id'],
                        'coords': coords.shape,
                        'dist': dist
                    }
                    if not dst_path.exists():
                        dst_path.parent.mkdir(exist_ok=True, parents=True)
                        img_crop.save(dst_path)
                except AssertionError as e:
                    print(f'Warning: {line_el.attrib["id"]}-{start_idx:02d} {e}')

with open(Path(args.output, 'data.json'), 'w') as f:
    json.dump(data, f)
