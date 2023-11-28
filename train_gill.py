"""Training example.

Example usage:
  python -u main.py \
      --dataset=cc3m  --val-dataset=cc3m \
      --opt-version='facebook/opt-6.7b' --visual-model='openai/clip-vit-large-patch14' \
      --exp_name='gill_exp'   --log-base-dir='runs/' \
      --batch-size=64  --val-batch-size=64  --precision='bf16'

Example run on 2 A6000 GPUs to reproduce the paper results:
  randport=$(shuf -i8000-9999 -n1)  # Generate a random port number
  python -u main.py \
      --dist-url "tcp://127.0.0.1:${randport}" --dist-backend 'nccl' \
      --multiprocessing-distributed --world-size 1 --rank 0 \
      --dataset=cc3m  --val-dataset=cc3m \
      --exp-name='gill_exp' --image-dir='data/'  --log-base-dir='runs/' \
      --precision='bf16'  --print-freq=100
"""
import argparse
from collections import OrderedDict
import json
import os
import random
import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
from warmup_scheduler import GradualWarmupScheduler
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
import torchvision
from transformers import AutoTokenizer

import wandb
import random

from gill import data
from gill import losses as losses_utils
from gill import models
from gill import utils
from gill import validate

llm_models = ['facebook/opt-125m', 'facebook/opt-350m', 'facebook/opt-1.3b', 'facebook/opt-2.7b',
              'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b']
datasets = ['cc3m']
best_gen_loss = 9999  # Variable to keep track of best model so far.


def list_of_str(arg):
    return list(map(str, arg.split(',')))


def parse_args(args):
    parser = argparse.ArgumentParser(description='GILL training')
    parser.add_argument('--llm_model', default='facebook/opt-6.7b',
                        help='OPT versions: ' +
                             ' | '.join(llm_models) +
                             ' (default: "facebook/opt-6.7b")')
    parser.add_argument('--visual-model', default='openai/clip-vit-large-patch14', type=str)
    parser.add_argument('--num-tokens', default=8, type=int, metavar='N', help='Number of [IMG] tokens to use.')
    parser.add_argument('--num-clip-tokens', default=77, type=int, metavar='N',
                        help='Number of CLIP token to use for generation.')

    parser.add_argument('--dataset', type=str, default="flintstones")

    parser.add_argument('--val-dataset', type=str, default="flintstones")
    parser.add_argument('--dataset-dir', default='../data/flintstones', type=str,
                        help='Dataset directory containing .tsv files.')
    parser.add_argument(
        "--clip_emb_file",
        type=str,
        default="clip_emb_text.pkl"
    )
    parser.add_argument('--image-dir', default='data/', type=str,
                        help='Dataset directory containing image folders.')
    parser.add_argument('--log-base-dir', default='./runs', type=str,
                        help='Base directory to write logs and ckpts to.')
    parser.add_argument('--exp-name', default='frozen', type=str,
                        help='Name of experiment, used for saving checkpoints.')
    parser.add_argument('--model-modes', type=list_of_str, default='generation')
    parser.add_argument('--interleave', action='store_true')

    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--steps_per_epoch', default=2000, type=int, metavar='N',
                        help='number of training steps per epoch')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--val_steps_per_epoch', default=-1, type=int, metavar='N',
                        help='number of validation steps per epoch')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 200), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--val-batch-size', default=64, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-warmup-steps', default=2000, type=int,
                        metavar='N', help='Number of steps to warm up lr.')
    parser.add_argument('--lr_schedule_step_size', default=5, type=int,
                        metavar='N', help='Number of steps before decaying lr.')
    parser.add_argument('--lr_schedule_gamma', default=0.1, type=float,
                        metavar='N', help='Decay parameter for learning rate scheduler.')
    parser.add_argument('--grad-accumulation-steps', default=1, type=int, metavar='N',
                        help='number of gradient accumulation steps')
    parser.add_argument('--grad-clip', default=1.0, type=float, help='gradient clipping amount')

    parser.add_argument('--precision', default='bf16', type=str, choices=['fp32', 'fp16', 'bf16'],
                        help="What precision to train in.")
    parser.add_argument('--cap-loss-scale', type=float, default=1.0, help="Scale on captioning loss.")
    parser.add_argument('--ret-loss-scale', type=float, default=1.0, help="Scale on retrieval loss.")
    parser.add_argument('--gen-loss-scale', type=float, default=1.0, help="Scale on retrieval loss.")
    parser.add_argument('--ce-loss-scale', type=float, default=0.5, help="Scale on retrieval loss.")

    parser.add_argument('--concat-captions-prob', type=float, default=0.5,
                        help="Probability of concatenating two examples sequentially for captioning.")
    parser.add_argument('--input-prompt', default='A picture of', type=str,
                        help="Input prompt for the language model, if any.")

    parser.add_argument('--image-size', default=224, type=int, metavar='N', help='Size of images.')
    parser.add_argument('--ret-emb-dim', default=256, type=int, metavar='N', help='Embedding dimension for retrieval.')
    parser.add_argument('--gen-emb-dim', default=768, type=int, metavar='N', help='Embedding dimension for generation.')

    text_fc_modes = ['linear', 'gill_mapper']
    parser.add_argument('--text-fc-mode', default='gill_mapper',
                        choices=text_fc_modes, help='What kind of translation mapping to use.')
    parser.add_argument('--ret-text-fc-mode', default='linear',
                        choices=text_fc_modes, help='What kind of translation mapping to use.')

    parser.add_argument('--max-len', default=32, type=int,
                        metavar='N', help='Maximum length to truncate captions / generations to.')
    parser.add_argument('--n-visual-tokens', default=4, type=int,
                        metavar='N', help='Number of visual tokens to use for the Frozen model.')

    parser.add_argument('--beta1', default=0.9, type=float, metavar='M',
                        help='beta1 for Adam')
    parser.add_argument('--beta2', default=0.95, type=float, metavar='M',
                        help='beta2 for Adam')
    parser.add_argument('--wd', '--weight-decay', default=0.01, type=float,
                        metavar='W', help='weight decay (default: 0.01)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:1337', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--wandb', action='store_true')
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    i = 1
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    while os.path.exists(args.log_dir):
        args.log_dir = os.path.join(args.log_base_dir, f'{args.exp_name}_{i}')
        i += 1
    os.makedirs(args.log_dir)

    with open(os.path.join(args.log_dir, f'args.json'), 'w') as wf:
        json.dump(vars(args), wf, indent=4)

    with open(os.path.join(args.log_dir, f'git_info.txt'), 'w') as wf:
        utils.dump_git_status(out_file=wf)

    print(f'Logging to {args.log_dir}.')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_gen_loss
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Create model
    model_args = models.GILLArgs()
    model_args.llm_model = args.llm_model
    model_args.visual_encoder = args.visual_model
    model_args.freeze_lm = True
    model_args.freeze_vm = True
    model_args.n_visual_tokens = args.n_visual_tokens
    model_args.ret_emb_dim = args.ret_emb_dim
    model_args.gen_emb_dim = args.gen_emb_dim
    model_args.text_fc_mode = args.text_fc_mode
    model_args.ret_text_fc_mode = args.ret_text_fc_mode
    model_args.num_tokens = args.num_tokens
    model_args.num_clip_tokens = args.num_clip_tokens
    model_args.text_emb_layers = [-1]
    model_args.max_len = args.max_len
    model_args.interleave = args.interleave
    assert args.num_tokens == 0 or 'gill_mapper' in model_args.text_fc_mode or (
                args.num_tokens * args.gen_emb_dim == args.num_clip_tokens * 768 or args.num_tokens * args.gen_emb_dim == args.num_clip_tokens * 1024), (
        f'{args.num_tokens} * {args.gen_emb_dim} != {args.num_clip_tokens} * 768 (or 1024)')

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, use_fast=False)
    if tokenizer.pad_token is None:
        if args.llm_model in ['EleutherAI/gpt-j-6B']:
            tokenizer.pad_token = tokenizer.eos_token
        elif args.llm_model.__contains__('llama'):
            tokenizer.pad_token = "$$"
        else:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        print("tokenizer.pad_token, tokenizer.eos_token:", tokenizer.pad_token, tokenizer.eos_token)
    # Add an image token for loss masking (and visualization) purposes.
    tokenizer.add_special_tokens({"cls_token": "<|image|>"})  # add special image token to tokenizer

    # Add [IMG] tokens to the vocabulary.
    model_args.retrieval_token_idx = []
    args.retrieval_token_idx = []
    for i in range(model_args.num_tokens):
        # print(f'Adding [IMG{i}] token to vocabulary.')
        # print(f'Before adding new token, tokenizer("[IMG{i}]") =', tokenizer(f'[IMG{i}]', add_special_tokens=False))
        num_added_tokens = tokenizer.add_tokens(f'[IMG{i}]')
        # print(f'After adding {num_added_tokens} new tokens, tokenizer("[IMG{i}]") =', tokenizer(f'[IMG{i}]', add_special_tokens=False))
        ret_token_idx = tokenizer(f'[IMG{i}]', add_special_tokens=False).input_ids
        assert len(ret_token_idx) == 1, ret_token_idx
        model_args.retrieval_token_idx.append(ret_token_idx[0])
        args.retrieval_token_idx.append(ret_token_idx[0])

    # Add [IMG] tokens to the vocabulary.
    model_args.gen_token_idx = model_args.retrieval_token_idx
    args.gen_token_idx = args.retrieval_token_idx

    # Save model args to disk.
    with open(os.path.join(args.log_dir, 'model_args.json'), 'w') as f:
        json.dump(vars(model_args), f, indent=4)

    # Data loading code
    if args.dataset == 'flintstones':
        train_dataset = data.StoryDataset(args, 'train', tokenizer)
        val_dataset = data.StoryDataset(args, 'val', tokenizer)
    else:
        train_dataset = data.PororoStoryDataset(args, 'train', tokenizer)
        val_dataset = data.PororoStoryDataset(args, 'val', tokenizer)
    print(f'Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples.')

    model = models.GILL(tokenizer, model_args)
    if args.precision == 'fp16':
        model = model.float()
    elif args.precision == 'bf16':
        model = model.bfloat16()

    # Print parameters and count of model.
    param_counts_text = utils.get_params_count_str(model)
    with open(os.path.join(args.log_dir, 'param_count.txt'), 'w') as f:
        f.write(param_counts_text)

    # Log trainable parameters to Tensorboard.
    _, total_trainable_params, total_nontrainable_params = utils.get_params_count(model)
    # writer = SummaryWriter(args.log_dir)
    # writer.add_scalar('params/total', total_trainable_params + total_nontrainable_params, 0)
    # writer.add_scalar('params/total_trainable', total_trainable_params, 0)
    # writer.add_scalar('params/total_non_trainable', total_nontrainable_params, 0)
    # writer.close()

    if args.wandb:
        wandb.init(project="gill", config=vars(args), entity="xiaoqian-shen", name=args.exp_name)
    else:
        wandb.init(project="gill", config=vars(args), entity="xiaoqian-shen", name=args.exp_name, mode="disabled")

    if not torch.cuda.is_available():
        print('WARNING: using CPU, this will be slow!')
        model = torch.nn.DataParallel(model)
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.val_batch_size = int((args.val_batch_size or args.batch_size) / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer_cls = torch.optim.AdamW
    print('Using torch.optim.AdamW as the optimizer.')
    optimizer = optimizer_cls(model.parameters(), args.lr,
                              betas=(args.beta1, args.beta2),
                              weight_decay=args.weight_decay,
                              eps=1e-8)

    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    scheduler_steplr = StepLR(optimizer, step_size=args.lr_schedule_step_size * args.steps_per_epoch,
                              gamma=args.lr_schedule_gamma)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=args.lr_warmup_steps,
                                       after_scheduler=scheduler_steplr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            state_dict = checkpoint['state_dict']
            try:
                model.load_state_dict(state_dict, strict=False)
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
                args.start_epoch = checkpoint['epoch']
                best_gen_loss = checkpoint['best_gen_loss']
                if args.gpu is not None:
                    best_gen_loss = best_gen_loss.to(args.gpu)
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            except:
                img_token_embeddings = state_dict['module.model.input_embeddings.weight'].detach()
                del state_dict['module.model.input_embeddings.weight']
                model.module.model.input_embeddings.weight.data[-args.num_tokens:, :].copy_(img_token_embeddings)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=(args.val_batch_size or args.batch_size), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate.validate(val_loader, model, tokenizer, criterion, epoch, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if epoch == 0:
            validate.validate(val_loader, model, tokenizer, criterion, epoch - 1, args)
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, tokenizer, criterion, optimizer, epoch, scheduler, args)

        # evaluate on validation set
        gen_loss = validate.validate(val_loader, model, tokenizer, criterion, epoch, args)

        wandb.log({"val_gen_loss": gen_loss})

        # remember best acc@1 and save checkpoint
        is_best = gen_loss < best_gen_loss
        best_gen_loss = min(gen_loss, best_gen_loss)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):

            # Only save non-frozen parameters.
            stripped_state_dict = {
                k: v for k, v in model.state_dict().items() if
                ('.lm' not in k and '.visual_model' not in k)
            }
            stripped_state_dict = OrderedDict(sorted(stripped_state_dict.items()))

            if is_best:
                state_dict = {}
                checkpoint = {}
                for k, v in stripped_state_dict.items():
                    state_dict[k.replace('module.', '')] = v.detach().clone()
                checkpoint['state_dict'] = state_dict
                finetuned_tokens = checkpoint['state_dict']['model.input_embeddings.weight'][-args.num_tokens:,
                                   :].detach().clone()
                checkpoint['state_dict']['model.input_embeddings.weight'] = finetuned_tokens

                with open(os.path.join(args.log_dir, 'pretrained_ckpt.pth.tar'), 'wb') as f:
                    torch.save(checkpoint, f)


def train(train_loader, model, tokenizer, criterion, optimizer, epoch, scheduler, args):
    ngpus_per_node = torch.cuda.device_count()
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    ce_losses = utils.AverageMeter('CeLoss', ':.4e')
    gen_losses = utils.AverageMeter('GenLoss', ':.4e')
    inp_emb_norm = utils.AverageMeter('TextEmbNorm', ':.4e')
    all_emb_norm = utils.AverageMeter('AllEmbNorm', ':.4e')
    ret_emb_norm = utils.AverageMeter('RetEmbNorm', ':.4e')

    progress = utils.ProgressMeter(
        args.steps_per_epoch,
        [batch_time, losses, ce_losses, gen_losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()

    for i, (_, images, captions, clip_emb) in enumerate(train_loader):
        actual_step = epoch * args.steps_per_epoch + i + 1
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            clip_emb = clip_emb.cuda(args.gpu, non_blocking=True)

        if args.precision == 'fp16':
            images = images.half()
        elif args.precision == 'bf16':
            images = images.bfloat16()

        model_modes = args.model_modes

        loss = 0

        for model_mode in model_modes:
            mode_start = time.time()

            (model_output, full_labels, last_embedding, _, visual_embs, visual_embs_norm,
             input_embs_norm, _, labels) = model(images, mode=model_mode, captions=captions)
            output = model_output.logits

            ce_loss = model_output.loss
            loss += ce_loss * args.ce_loss_scale
            ce_losses.update(ce_loss.item(), len(images))

            wandb.log({"ce_loss": ce_loss})

            if args.num_tokens != 0 and args.num_clip_tokens != args.num_tokens:
                seq_len = clip_emb.shape[1]
                last_embedding = last_embedding.reshape((last_embedding.shape[0], seq_len, -1))
                assert last_embedding.shape == clip_emb.shape, (last_embedding.shape == clip_emb.shape)

            image_loss = losses_utils.l2_loss(clip_emb, last_embedding)  # (N,)
            gen_loss = args.gen_loss_scale * image_loss.mean()
            loss += gen_loss
            gen_losses.update(gen_loss.item(), len(images))

            wandb.log({"gen_loss": gen_loss})

            inp_emb_norm.update(input_embs_norm.item(), len(images))

        loss = loss / args.grad_accumulation_steps
        losses.update(loss.item(), len(images))
        loss.backward()

        # Update weights
        if ((i + 1) % args.grad_accumulation_steps == 0) or (i == args.steps_per_epoch - 1):
            # Zero out gradients of the embedding matrix outside [IMG].
            for param in model.module.model.input_embeddings.parameters():
                assert param.grad.shape[0] == len(tokenizer)
                # Keep other embeddings frozen.
                mask = torch.zeros((param.grad.shape[0], 1)).to(param.grad)
                for ret_idx in args.retrieval_token_idx:
                    mask[ret_idx] = 1
                for gen_idx in args.gen_token_idx:
                    mask[gen_idx] = 1
                param.grad = param.grad * mask

            # compute gradient and do SGD step
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            # Normalize trainable embeddings.
            frozen_norm = torch.norm(model.module.model.input_embeddings.weight[:-args.num_tokens, :], dim=1).mean(0)
            for ret_idx in args.retrieval_token_idx:
                trainable_weight = model.module.model.input_embeddings.weight[ret_idx, :]
                model.module.model.input_embeddings.weight[ret_idx, :].div_(trainable_weight.norm(dim=-1) / frozen_norm)

            # Log norms to Tensorboard.
            embedding_norm = torch.norm(model.module.model.input_embeddings.weight, dim=1).mean()
            ret_embedding_norm = torch.norm(model.module.model.input_embeddings.weight[args.retrieval_token_idx, :],
                                            dim=-1).mean()
            all_emb_norm.update(embedding_norm.item(), len(images))
            ret_emb_norm.update(ret_embedding_norm.item(), len(images))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if actual_step == 1 or (i + 1) % args.print_freq == 0:

            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                inp_emb_norm.all_reduce()
                ret_time.all_reduce()
                all_emb_norm.all_reduce()
                ret_emb_norm.all_reduce()
                gen_losses.all_reduce()

            progress.display(i + 1)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                # Append caption text.
                pred_tokens = output[:, args.n_visual_tokens - 1:-1, :].argmax(dim=-1)
                generated_captions = tokenizer.batch_decode(pred_tokens, skip_special_tokens=False)

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            inp_emb_norm.reset()
            all_emb_norm.reset()
            ret_emb_norm.reset()
            gen_losses.reset()

        if i == args.steps_per_epoch - 1:
            break

        scheduler.step()
        curr_lr = scheduler.get_last_lr()
        wandb.log({"lr": curr_lr[0]})

# Disable tokenizer parallelism.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == '__main__':
    main(sys.argv[1:])
