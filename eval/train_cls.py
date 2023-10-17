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
    def __init__(self, root_path, im_input_size, mode='train'):
        self.root_path = root_path
        with open(os.path.join(self.root_path, 'labels_bg.pkl'),'rb') as f:
            self.labels = pickle.load(f)['label']
        with open(os.path.join(self.root_path, 'split.json'),'r') as f:
            splits = json.load(f)
        self.splits = splits
        train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
        self.mode = mode
        if mode == 'train':
            self.ids = train_ids + val_ids + test_ids
            self.transform = transforms.Compose([
                transforms.Resize(im_input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.ids = test_ids
            self.transform = transforms.Compose([
                transforms.Resize(im_input_size),
                transforms.CenterCrop(im_input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        video_meta_path = os.path.join(self.root_path, 'video_frames_sampled_4x', f'{img_id}.npy')
        video_array = np.load(video_meta_path)
        n_frames = video_array.shape[0]
        random_range = random.randrange(n_frames)
        image = Image.fromarray(video_array[random_range])
        label = self.labels[img_id]
        return self.transform(image), torch.Tensor(label)

    def __len__(self):
        return len(self.ids)

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, is_inception=False):

    since = time.time()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            total_gt_positives = 0
            total_pred_positives = 0
            total_true_positives = 0
            total_frame_positives = 0
            total_char_positives = np.zeros(args.num_classes)

            # Iterate over data.
            for i, (inputs, labels) in tqdm(enumerate(dataloaders[phase])):

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    preds = torch.round(nn.functional.sigmoid(outputs))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                # iter_true_positives = torch.sum(preds == labels.data)
                xidxs, yidxs = torch.where(labels.data == 1)
                iter_true_positives = sum(
                    [labels.detach().cpu().numpy()[xidx, yidx] == preds.detach().cpu().numpy()[xidx, yidx] for
                     xidx, yidx in zip(xidxs, yidxs)])
                iter_pred_positives = torch.sum(preds)
                iter_gt_positives = xidxs.size(0)

                iter_frame_positives =0
                for l, p in zip(preds.detach().cpu().numpy(), labels.detach().cpu().numpy()):
                    if np.array_equal(l, p):
                        iter_frame_positives += 1
                total_frame_positives += iter_frame_positives
                total_char_positives = total_char_positives + np.sum(np.equal(preds.detach().cpu().numpy(), labels.detach().cpu().numpy()), axis=0)

                running_loss += loss.item() * inputs.size(0)
                total_pred_positives += iter_pred_positives
                total_gt_positives += iter_gt_positives
                total_true_positives += iter_true_positives

                if phase == 'train' and i%100 == 0:
                    r = total_true_positives*100.0/total_gt_positives
                    p = total_true_positives*100.0/total_pred_positives
                    f1 = 2*r*p/(r+p)
                    print('Phase %s; Steps %s/%s Loss: %.4f Frame Acc.: %.4f Recall: %.4f Precision: %.4f F-Score: %.4f' %
                          (phase, i, len(dataloaders[phase]), running_loss/(args.batch_size*(i+1)),
                           total_frame_positives*100/(args.batch_size*(i+1)), r, p, f1))
                    print("Accuracy by Class: ", total_char_positives/(args.batch_size*(i+1)))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_recall = total_true_positives*100.0/total_gt_positives
            epoch_precision = total_true_positives * 100.0 / total_pred_positives
            epoch_f1_score = 2*epoch_recall*epoch_precision/(epoch_recall+epoch_precision)
            epoch_frame_acc = total_frame_positives * 100 / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Frame Acc: {:.4f} Recall: {:.4f} Precision: {:.4f} F1 Score: {:.4f}'.format(phase, epoch_loss, epoch_frame_acc,
                                                                                                         epoch_recall, epoch_precision, epoch_f1_score))
            print("Accuracy by Class: ", total_char_positives / len(dataloaders[phase].dataset))

            # deep copy the model
            if phase == 'val':
                val_acc_history.append(epoch_frame_acc)
                with open('results.txt', 'a+') as f:
                    f.write("Epoch %s\n" % epoch)
                    f.write(
                        '{} Loss: {:.4f} Frame Acc: {:.4f} Recall: {:.4f} Precision: {:.4f} F1 Score: {:.4f}'.format(
                            phase, epoch_loss, epoch_frame_acc,
                            epoch_recall, epoch_precision, epoch_f1_score))
                    f.write("\n")
                    f.write("Accuracy by Class: " + " ".join(
                        [str(a) for a in total_char_positives / len(dataloaders[phase].dataset)]))
                    f.write("\n\n")
                if epoch_frame_acc > best_acc:
                    best_acc = epoch_frame_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            if phase =='train':
                torch.save(model.state_dict(), os.path.join(args.save_path, 'epoch-%s.pt' % epoch))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
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
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
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
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def main(args):

    args.save_path = args.save_path + '_' + str(args.batch_size) + '_' + str(args.learning_rate)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with open('results.txt', 'a+') as f:
        f.write("Logging for %s" % args.save_path)
        f.write("\n-----------------------------------------\n")

    print("Initializing model")
    # Initialize the model for this run
    model_ft, input_size = initialize_model(args.model_name, args.num_classes, args.feature_extract, use_pretrained=True)
    # Print the model we just instantiated
    # print(model_ft)
    print("Initializing Datasets and Dataloaders...")
    # Create training and validation datasets
    image_datasets = {x: ImageDataset(args.data_dir, input_size, x) for x in ['train', 'val']}
    print(image_datasets['train'][0])
    # Data augmentation and normalization for training
    # Just normalization for validation
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if args.feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(params_to_update, lr=args.learning_rate)

    # Setup the loss fxn
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    # Train and evaluate
    _, _ = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
                                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                                 num_epochs=args.num_epochs, is_inception=(args.model_name=="inception"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train for Character Recall & InceptionScore')
    parser.add_argument('--data_dir',  type=str, default='/ibex/ai/home/shenx/story/data/flintstones')
    parser.add_argument('--model_name', type=str, default='inception')
    parser.add_argument('--save_path', type=str, default='/ibex/ai/home/shenx/story/diffstory/eval/classifier')
    parser.add_argument('--num_classes', type=int, default=323, help='7 for char, 323 for bg')
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--feature_extract', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    args = parser.parse_args()

    main(args)