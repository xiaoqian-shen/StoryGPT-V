import torchvision.transforms as transforms
import argparse
import os
from fid_score import fid_score

import torch
import json
import numpy as np
import random
import os
from PIL import Image

class EvalData(torch.utils.data.Dataset):
    def __init__(self, root_path, gen_path, is_gen=False):
        self.root_path = root_path
        with open(os.path.join(self.root_path, 'split.json'),'r') as f:
            self.image_ids = json.load(f)['test']
        with open(os.path.join(self.root_path, 'cleaned_annotations.json'),'r') as f:
            self.ann_file = json.load(f)
        self.gen_path = gen_path
        self.is_gen = is_gen
        gen_ids=[filename.split('.')[0] for filename in os.listdir(self.gen_path)]
        # self.image_ids = list(set(self.image_ids).intersection(set(gen_ids)))
        self.image_ids = gen_ids
        self.transforms = transforms.Compose([
            transforms.Resize((args.imsize, args.imsize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        if not self.is_gen:
            image_id = self.image_ids[idx]
            info = self.ann_file[image_id]
            video_meta_path = os.path.join(self.root_path, 'video_frames_sampled_4x', f'{image_id}.npy')
            video_array = np.load(video_meta_path)
            n_frames = video_array.shape[0]
            random_range = random.randrange(n_frames)
            image = Image.fromarray(video_array[random_range])
            return self.transforms(image).unsqueeze(0)
            # return self.transforms(Image.fromarray(video_array))
        else:
            image_id = self.image_ids[idx]
            image_path = os.path.join(self.gen_path, f'{image_id}.png')
            image = Image.open(image_path)
            return self.transforms(image).unsqueeze(0)

def main(args):
    
    ref_dataset = EvalData(root_path=args.ref_dir, gen_path=args.gen_dir)
    gen_dataset = EvalData(root_path=args.ref_dir, gen_path=args.gen_dir, is_gen=True)

    fid = fid_score(ref_dataset, gen_dataset, cuda=True, normalize=True, r_cache=os.path.join(args.ref_dir, 'fid_cache_%s.npz' % args.mode), batch_size=16)
    exp_name = args.gen_dir.split('/')[-1]
    if args.test:
        print()
        print('FID: ', fid, flush=True)
        print()
    else:
        import json

        with open('/ibex/ai/home/shenx/story/diffstory/eval.json', 'r') as f:
            result_dict = json.load(f)

        result_dict[exp_name] = [0, 0, 0, 0, round(fid, 2)]

        with open('/ibex/ai/home/shenx/story/diffstory/eval.json', 'w') as f:
            json.dump(result_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Frechet Story and Image distance')
    parser.add_argument('--gen_dir', type=str, required=True)
    parser.add_argument('--root_path', type=str, default='/ibex/ai/home/shenx/story/data')
    parser.add_argument('--dataset', type=str, default='flintstones')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--imsize', type=int, default=512)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    args.ref_dir = os.path.join(args.root_path, args.dataset)
    print(args.gen_dir)
    main(args)
