from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import time
import os
import copy
import numpy as np
import argparse

import os, re
import torch.utils.data
from torchvision import transforms
from PIL import Image
import json
import pickle
import random

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, im_input_size, gen_path, num_classes):
        self.root_path = root_path
        if num_classes == 323:  # bg
            with open(os.path.join(self.root_path, 'labels_bg.pkl'), 'rb') as f:
                self.labels = pickle.load(f)['label']
        else: # character
            with open(os.path.join(self.root_path, 'labels.pkl'), 'rb') as f:
                self.labels = pickle.load(f)
            
        with open(os.path.join(self.root_path, 'split.json'), 'r') as f:
            splits = json.load(f)
        self.splits = splits
        self.gen_path = gen_path
        self.ids = splits["test"]
        # self.ids = list(set(self.ids).intersection(set([filename.split('.')[0] for filename in os.listdir(self.gen_path)])))
        self.ids = [filename.split('.')[0] for filename in os.listdir(self.gen_path)]
        self.transform = transform = transforms.Compose([
            transforms.Resize(im_input_size),
            transforms.CenterCrop(im_input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        if self.gen_path is None:
            img_id = self.ids[idx]
            video_meta_path = os.path.join(self.root_path, 'video_frames_sampled_4x', f'{img_id}.npy')
            video_array = np.load(video_meta_path)
            n_frames = video_array.shape[0]
            random_range = random.randrange(n_frames)
            image = Image.fromarray(video_array[random_range])
            label = self.labels[img_id]
            return self.transform(image), torch.Tensor(label)
        else:
            img_id = self.ids[idx]
            image_path = os.path.join(self.gen_path, f'{img_id}.png')
            image = Image.open(image_path)
            label = self.labels[img_id]
            return self.transform(image), torch.Tensor(label)

    def __len__(self):
        return len(self.ids)


def eval_model(model, dataloaders, device):
    model.eval()  # Set model to evaluate mode

    total_frames = 0
    total_gt_positives = 0
    total_pred_positives = 0
    total_true_positives = 0
    total_frame_positives = 0
    total_char_positives = np.zeros(model.num_classes)

    for inputs, labels in dataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)

        total_frames += inputs.shape[0]

        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.round(nn.functional.sigmoid(outputs))

        xidxs, yidxs = torch.where(labels.data == 1)
        iter_true_positives = sum(
            [labels.detach().cpu().numpy()[xidx, yidx] == preds.detach().cpu().numpy()[xidx, yidx] for
             xidx, yidx in zip(xidxs, yidxs)])
        iter_pred_positives = torch.sum(preds)
        iter_gt_positives = xidxs.size(0)

        iter_frame_positives = 0
        for l, p in zip(preds.detach().cpu().numpy(), labels.detach().cpu().numpy()):
            if np.array_equal(l, p):
                iter_frame_positives += 1
        total_frame_positives += iter_frame_positives
        total_char_positives = total_char_positives + np.sum(
            np.equal(preds.detach().cpu().numpy(), labels.detach().cpu().numpy()), axis=0)

        total_pred_positives += iter_pred_positives
        total_gt_positives += iter_gt_positives
        total_true_positives += iter_true_positives

    epoch_recall = total_true_positives * 100.0 / total_gt_positives
    epoch_precision = total_true_positives * 100.0 / total_pred_positives
    epoch_f1_score = 2 * epoch_recall * epoch_precision / (epoch_recall + epoch_precision)
    epoch_frame_acc = total_frame_positives * 100 / total_frames
    print("Accuracy by Class: ", total_char_positives / total_frames)
    return epoch_frame_acc, epoch_f1_score


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet101":
        """ Resnet50
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.num_classes = num_classes
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def main(args):
    model_ft, input_size = initialize_model(args.model_name, args.num_classes, args.feature_extract,
                                            use_pretrained=True)
    print("Initializing Datasets and Dataloaders...")
    # Create training and validation datasets
    image_datasets = ImageDataset(args.data_dir, input_size, args.gen_dir, args.num_classes)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)

    model_ft.load_state_dict(torch.load(args.model_path))

    for param in model_ft.parameters():
        param.requires_grad = False

    acc, f1 = eval_model(model_ft, dataloaders, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    exp_name = args.gen_dir.split('/')[-1]

    if args.test:
        print()
        print(f'{args.mode} Acc: ', acc, flush=True)
        print(f'{args.mode} F1: ', f1, flush=True)
        print()
    else:
        import json

        with open('/ibex/ai/home/shenx/story/diffstory/eval.json', 'r') as f:
            result_dict = json.load(f)

        if args.num_classes == 7:
            result_dict[exp_name][0] = round(acc, 2)
            result_dict[exp_name][1] = round(float(torch.round(f1, decimals=2).cpu().numpy()), 2)
            print(result_dict[exp_name], flush=True)
            with open('/ibex/ai/home/shenx/story/diffstory/eval.json', 'w') as f:
                json.dump(result_dict, f)
        else:
            result_dict[exp_name][2] = round(acc, 2)
            result_dict[exp_name][3] = round(float(torch.round(f1, decimals=2).cpu().numpy()), 2)
            print(result_dict[exp_name], flush=True)
            with open('/ibex/ai/home/shenx/story/diffstory/eval.json', 'w') as f:
                json.dump(result_dict, f)
            import json
            import csv
            with open('/ibex/ai/home/shenx/story/diffstory/eval.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['exp', 'text_only', 'coref', 'steps', 'char-acc', 'char-f1', 'bg-acc', 'bg-f1', 'fid'])
                for key, values in result_dict.items():
                    output_name = key.split('-')[-1]
                    exp = '-'.join(key.split('-')[:-1])
                    text_only = True if 'textonly' in output_name else False
                    coref = True if 'coref' in key else False
                    steps = output_name.split('_')[0]
                    writer.writerow([exp, text_only, coref, steps, ] + values)
            # if (result_dict[exp_name][0] < 86.5 and not coref) or result_dict[exp_name][0] < 69:
            #     os.system(
            #         f'rm -r /ibex/project/c2133/aa_shenx/diffstory/checkpoints/stable-diffusion-v1-5/flintstones/{exp}/checkpoint-{steps}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train for Character Recall & InceptionScore')
    parser.add_argument('--root_path', type=str, default='/ibex/ai/home/shenx/story/data')
    parser.add_argument('--dataset', type=str, default='flintstones')
    parser.add_argument('--model_name', type=str, default='inception')
    parser.add_argument('--gen_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, default='char', help='char or bg')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--feature_extract', type=bool, default=False)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    args.data_dir = os.path.join(args.root_path, args.dataset)
    if args.mode == 'char':
        args.model_path = os.path.join(args.root_path, args.dataset, 'classifier_char.pt')
        args.num_classes = 7 if args.dataset == 'flintstones' else 9
    else:
        args.model_path = os.path.join(args.root_path, args.dataset, 'classifier_bg.pt')
        args.num_classes = 323
    main(args)
