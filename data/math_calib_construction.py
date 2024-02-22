import json
import fire
import random
import os.path as osp
from glob import glob
from tqdm import tqdm


def main(dataset_path: str, output_path: str = './', seed: int = 42):
    random.seed(seed)
    data_all = []
    for filename in tqdm(glob(f'{dataset_path}/MATH/train/*/*')):
        data = json.load(open(filename, 'r'))
        it = data['problem'] + ' ' + data['solution']
        data_all.append({'text': it})

    random.shuffle(data_all)
    with open(osp.join(output_path, 'math_pretrain_style.json'), 'w') as f:
        json.dump(data_all, f)


if __name__ == '__main__':
    fire.Fire(main)
