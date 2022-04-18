import wandb
from sqlalchemy import false, true
from util import *

import argparse
import os
import random
import shutil
import time
from tqdm import tqdm
import typing as t
import numpy as np
import pandas as pd

import torch
from torch import device, nn, Tensor
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import dataset.custom_imagefolder as custom_imagefolder

import torchvision.models as models
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

os.environ["WANDB_MODE"] = "offline"
wandb.login()


def make(config):

    print("=> using pre-trained model '{}'".format(config.arch))
    model = models.__dict__[config.arch]()
    model.load_state_dict(torch.load(
        './logs/pretrained/{}.pth'.format(config.arch), map_location=config.device))

    classifier = nn.Linear(in_features=model.fc.in_features,
                           out_features=model.fc.out_features)
    model.fc = nn.Identity()

    model = model.to(config.device)
    classifier.to(config.device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(config.device)
    optimizer = torch.optim.SGD([classifier.parameters(), model.parameters()], config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
    
    cudnn.benchmark = True

    return model, classifier, criterion, optimizer


def caltech_imagenet_dataloader(config):
    source_trainval_ratio = 0.8
    class_list = np.asarray(pd.read_csv(
        config.classlist_path, skiprows=[0], header=None), dtype=int)
    source_class_list = np.unique(class_list[:, 0])
    target_class_list = np.unique(class_list[:, 1])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    source_traindir = os.path.join(config.source_image_path, 'train')
    source_valdir = os.path.join(config.source_image_path, 'val')

    source_train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(source_traindir, train_transform),
        batch_size=config.batch_size, shuffle=True,
        num_workers=config.workers, pin_memory=True)
    source_val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(source_valdir, eval_transform),
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True)

    target_train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(config.target_image_path, train_transform),
        batch_size=config.batch_size, shuffle=True,
        num_workers=config.workers, pin_memory=True)
    
    # 84 common classes with imagenet and map target to imagenet index 
    target_test_loader = torch.utils.data.DataLoader(
        custom_imagefolder.ImageFolder_classlist(
            config.target_image_path, eval_transform, classlist=target_class_list, targetmap=source_class_list),
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True)

    return source_train_loader, source_val_loader, target_train_loader, target_test_loader


def main_pipeline(*, args):
    with wandb.init(project=args.title, allow_val_change=True, config=args):
        config = wandb.config

        global best_acc1
        if config.gpu is not None:
            print("Use GPU: {} for training".format(config.gpu))
        model, classifier, criterion, optimizer = make(config)

        # Data loading code
        source_train_loader, source_val_loader, target_train_loader, target_test_loader = caltech_imagenet_dataloader(
            config)

        if config.evaluate:
            validate(source_val_loader, model, classifier, criterion, config)
            return

        # record target accuracy before adaption
        validate(target_test_loader, model, classifier, criterion, config)

        # Tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(model, criterion, log="none", log_freq=10)
        for epoch in tqdm(range(config.start_epoch, config.epochs)):
            adjust_learning_rate(optimizer, epoch, config)

            # train for one epoch
            train_progress = train(
                source_train_loader, target_train_loader, model, classifier, criterion, optimizer, epoch, config)

            # evaluate on validation set
            acc1, val_progress = validate(
                source_val_loader, model, criterion, config)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            save_checkpoint({
                'arch': config.arch,
                'model_state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict()
            }, is_best, config.dir)

            d = {}
            for meter in train_progress.meters:
                d[meter.name] = meter.val
                d[meter.name + '_avg'] = meter.avg
            for meter in val_progress.meters:
                d[meter.name] = meter.val
                d[meter.name + '_avg'] = meter.avg

            wandb.log(d, step=epoch)

        # record target accuracy after adaption
        validate(target_test_loader, model, classifier, criterion, config)


def train(sourcetrain_loader, target_train_loader, model, classifier, criterion, optimizer, epoch, config):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    cls_losses = AverageMeter('Loss', ':.4e')
    reg_losses = AverageMeter('Loss', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    intra_distance = AverageMeter('intra_distance', ':6.2f')
    inter_distance = AverageMeter('inter_distance', ':6.2f')
    progress = ProgressMeter(
        len(sourcetrain_loader),
        [batch_time, data_time, cls_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    classifier.train()

    end = time.time()
    for i, (source_images, source_label) in enumerate(sourcetrain_loader):
        target_loader_iter = iter(target_train_loader)
        # Get the data from the base dataset
        try:
            target_image, _ = target_loader_iter.next()
        except StopIteration:
            target_loader_iter = iter(target_train_loader)
            target_image, _ = target_loader_iter.next()

        # measure data loading time
        data_time.update(time.time() - end)

        source_images.to(config.device)
        source_label.to(config.device)
        target_image.to(config.device)

        # compute output
        source_features_, target_features_ = torch.split(model(torch.cat([source_images, target_image], dim=0)),
                                                         [source_images.size()[0], target_image.size()[0]], dim=0)
        output = classifier(source_features_)
        cl_loss = criterion(output, source_label)

        reg_loss, meta = regularization(
            source_features_, target_features_, sigma=config.sigma)
        intra_distance.update(meta["minimum_intra_nearest_distance"])
        inter_distance.update(meta["minimum_inter_nearest_distance"])

        loss = (cl_loss + config.reg_weight * reg_loss)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, source_label, topk=(1, 5))
        cls_losses.update(cl_loss.item(), source_images.size(0))
        reg_losses.update(reg_loss.item(), source_images.size(0))
        losses.update(loss.item(), source_images.size(0))
        top1.update(acc1[0], source_images.size(0))
        top5.update(acc5[0], source_images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.display(i)

    return progress


def validate(val_loader, model, classifier, criterion, config):
    batch_time = AverageMeter('val_Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('val_Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('val_Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('val_Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    classifier.eval()
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if config.gpu is not None:
                images = images.cuda(config.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(config.gpu, non_blocking=True)

            # compute output
            output = model(images)
            output = classifier(output)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                progress.display(i)

        progress.display_summary()

    return top1.avg, progress


def regularization(*, source_feature: Tensor, target_feature: Tensor, sigma: float = 0.8, device) -> t.Tuple[Tensor, t.Dict]:
    def _regularize(source_feature, target_feature):
        source_size = source_feature.size()[0]
        target_size = target_feature.size()[0]
        all_features = torch.cat([source_feature, target_feature], dim=0)
        squared_features = torch.cdist(all_features, all_features, p=2) + (
            torch.eye(all_features.size()[0]).to(device) * 1e5)
        distance_map = torch.exp(-squared_features / (2 * sigma ** 2))
        distance_map = distance_map * (1 - torch.eye(all_features.size()[0]))
        intra_domain_distance_map = distance_map[:source_size, :source_size]
        intra_nominator = torch.max(intra_domain_distance_map, dim=1)[0]
        # intra_denominator = torch.sum(intra_domain_distance_map, dim=1)
        # intra_domain_distance_map = intra_nominator / intra_denominator
        source_source_nearest_neighbor_distance_map = squared_features[:source_size, :source_size]
        source_source_nearest_neighbor_distances = source_source_nearest_neighbor_distance_map.min(dim=1)[
            0]

        inter_domain_distance_map = distance_map[:source_size, source_size:]
        inter_nominator = torch.max(inter_domain_distance_map, dim=1)[0]
        # inter_denominator = torch.sum(inter_domain_distance_map, dim=1)
        # inter_domain_distance_map = inter_nominator / inter_denominator
        source_target_nearest_neighbor_distance_map = squared_features[:source_size, source_size:]
        source_target_nearest_neighbor_distances = source_target_nearest_neighbor_distance_map.min(dim=1)[
            0]

        meta_info = {
            "minimum_intra_nearest_distance": source_source_nearest_neighbor_distances.mean().item(),
            "minimum_inter_nearest_distance": source_target_nearest_neighbor_distances.mean().item()
        }

        return torch.stack([intra_nominator, inter_nominator], dim=1).softmax(1), meta_info

    p1, meta1 = _regularize(source_feature, target_feature)
    p2, meta2 = _regularize(target_feature, source_feature)

    meta = {}
    for key in meta1.keys():
        meta[key] = (meta1[key] + meta2[key]) / 2

    return -entropy(p1) - entropy(p2), meta


def entropy(p):
    return torch.sum(-p * torch.log(p), dim=1).mean()


# Initialize
parser = argparse.ArgumentParser(description='Nearest Prior Training Pipeline')
parser.add_argument('--title', metavar='title',
                    help='title of this run')
parser.add_argument('--source_image_path', metavar='source_image_path', default='./dataset/ILSVRC/Data/CLS-LOC',
                    help='path to cource images folder')
parser.add_argument('--target_image_path', metavar='target_image_path', default='./dataset/256_objectcategories/256_ObjectCategories/',
                    help='path to target images folder')
parser.add_argument('--classlist_path', metavar='classlist_path', default='./dataset/imagenet_to_caltech.csv',
                    help='path to class list')
parser.add_argument('--checkpoint_dir', type=str, default='./logs/ImageNet_caltech256/',
                    help='directory to save the checkpoints')
parser.add_argument('--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet50)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--workers', default=8, type=int,
                    help='number of workers.')
parser.add_argument('--sigma', default=0.8, type=float,
                    help='sigma.')
parser.add_argument('--reg_weight', default=0.05, type=float,
                    help='Regularization loos weight.')

args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

if torch.cuda.is_available():
    dev = "cuda:{0}".format(args.gpu)
else:
    dev = "cpu"
args.device = torch.device(dev)

main_pipeline(args=args)
