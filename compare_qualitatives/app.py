import argparse
from flask import Flask, render_template, send_file
from pathlib import Path
from collections import defaultdict

app = Flask(__name__)
parser = argparse.ArgumentParser(description='Flask app with argparse')
parser.add_argument('paths', metavar='path', type=str, nargs='+', help='list of paths')
args = parser.parse_args()


class ImgSaver:
    def __init__(self):
        self.id_to_img = {}
        self.img_to_id = {}
    
    def save(self, img_path):
        img_id = len(self.id_to_img)
        self.id_to_img[img_id] = img_path
        self.img_to_id[img_path] = img_id
        return img_id

saver = ImgSaver()
data = defaultdict(list)

paths = [Path(path) for path in args.paths]


# for path in paths:
#     if 'iam' not in path.stem:
#         continue
#     for img in path.rglob('*.png'):
#         img_id = int(img.stem)
#         author_id = img.parent.stem
#         dst_img = path / author_id / f'test_{author_id}_{img_id:04d}.png'
#         # rename
#         img.rename(dst_img)

for img in sorted(paths[0].rglob('*.png')):
    img_paths = [path / img.relative_to(paths[0]) for path in paths]
    assert all([p.exists() for p in img_paths])
    img_ids = [saver.save(p) for p in img_paths]
    img_urls = [f'/image/{img_id}' for img_id in img_ids]
    data[img.parent.name].append(img_urls)

@app.route('/')
def index():
    return render_template('index.html', data=data, paths=['/'.join(p.parts[-2:]) for p in paths])

@app.route('/<author_id>')
def author(author_id):
    new_data = {author_id: data[author_id]}
    return render_template('index.html', data=new_data, paths=['/'.join(p.parts[-2:]) for p in paths])

@app.route('/image/<image_id>')
def get_image(image_id):
    image_path = saver.id_to_img[int(image_id)]
    return send_file(image_path, mimetype='image/png')

if __name__ == '__main__':
    script_dir = Path(__file__).resolve().parent
    template_dir = script_dir / 'templates'
    app.template_folder = template_dir

    # Start Flask app
    app.run(debug=True)