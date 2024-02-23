import gzip
import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


if __name__ == '__main__':
    from save_db_for_hwd import setup_loader
    from train import add_arguments, set_seed

    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    parser.add_argument('--dst', type=Path, default='files/iam')
    args = parser.parse_args()
    
    with gzip.open('files/iam_htg_setting.json.gz', 'rt', encoding='utf-8') as file:
        data = json.load(file)
    data = [d for d in data if d['dst'].startswith('test')]

    set_seed(args.seed)

    loader = setup_loader(0, args)

    old_data = defaultdict(list)
    for el in tqdm(data):
        old_data[Path(el['dst']).parts[1]].append(el)

    new_data = []
    for sample in tqdm(loader):
        for lbl, author in zip(sample['style_text'], sample['style_author']):
            el = old_data[author].pop(0)
            el['word'] = lbl
            new_data.append(el)
    
    with gzip.open('files/iam_lines_htg_setting.json.gz', 'wt', encoding='utf-8') as file:
        json.dump(new_data, file, ensure_ascii=False, indent=2)
    
    print('Done')