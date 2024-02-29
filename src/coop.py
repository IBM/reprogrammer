import os
import time
import argparse

parser = argparse.ArgumentParser(description='PyTorch CLIP CoOp Tuning')
parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
parser.add_argument('--in-dataset', default="ImageNet-1k", type=str, help='in-distribution dataset')
parser.add_argument('--model-arch', default='ViT-B/32', type=str, help='model architecture')
parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--save-epoch', default=50, type=int, help='save the model every save_epoch')

parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float, help='initial learning rate for image encoder')
parser.add_argument('--wd', '--weight-decay',default=0.0005, type=float, help='weight decay for text encoder (default: 0.1)')
parser.add_argument('--no-augment', dest='augment', action='store_false', help='whether to use standard augmentation (default: False)')
parser.add_argument("--warmup_length", type=int, default=500, help='initial warmup iterations for the scheduler (default: 500)')

parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate for the image reprogramming perturbation (default: 0.0)')
parser.add_argument('--image-resolution', default=224, type=int, help='preprocessing image resolution (default: 64)')
parser.add_argument('--print-freq', '-p', default=150, type=int, help='print frequency (default: 150)')
parser.add_argument('--random-seed', default=1, type=int, help='The seed used for torch & numpy')
parser.add_argument('--load', default=None, type=str, help='name of experiment being loaded')
parser.add_argument('--name', required=True, type=str, help='name of experiment')
parser.set_defaults(augment=True)

# Grabbing cli arguments and printing results
args = parser.parse_args()
print_args = '*'*45
for key,value in args._get_kwargs():
    print_args = print_args + '\n- ' + str(key) + " -> " + str(value)

print_args = print_args + '\n' + '*'*45
print(print_args)

directory = "/nobackup/ageng/checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
if not os.path.exists(directory):
    os.makedirs(directory)

save_state_file = os.path.join(directory, 'args.txt')
fw = open(save_state_file, 'w')
print(print_args, file=fw)
fw.close()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import clip
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from clip.coop_model import CoOpCLIP
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils import ImageNet, cosine_lr
from utils.label_map import cifar10_labels, cifar10_classnames, imagenet_classnames

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)


def main():
    # Setting up data augmentation if needed
    train_transform = transforms.Compose([
        transforms.Resize(args.image_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_resolution),
        convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    test_transform = transforms.Compose([
        transforms.Resize(args.image_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        convert_image_to_rgb, transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    # Setting up testing dataset
    kwargs = {'num_workers': 1, 'pin_memory': True}
    if args.in_dataset == "CIFAR-10":
        classnames = cifar10_classnames
        labels = cifar10_labels
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('/nobackup/agd/datasets/cifar10', train=True, download=True, transform=train_transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('/nobackup/agd/datasets/cifar10', train=False, transform=test_transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.in_dataset == 'ImageNet-1k':
        classnames = imagenet_classnames
        labels = cifar10_labels
        train_loader = torch.utils.data.DataLoader(
            ImageNet(root='/nobackup/ageng/datasets/ImageNet-1k/', train=True, transform=train_transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            ImageNet('/nobackup/ageng/datasets/ImageNet-1k/', train=False, transform=test_transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    
    # Loading zero-shot CLIP model
    clip_model, preprocess = clip.load('ViT-B/32')

    # Issue -> https://github.com/openai/CLIP/issues/40
    clip_model = clip_model.float().cpu()

    # Specifying gpu usage for model
    model = CoOpCLIP(classnames, clip_model)
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Defining image and text loss function 
    ce_loss = torch.nn.CrossEntropyLoss().cuda()
    
    # Defining SGD optimizers
    params = [p for n, p in list(model.named_parameters()) if 'prompt_learner.ctx' in n]
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=False)

    # Defining learning rate scheduler
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs*len(train_loader))

    # Iterating through training epochs
    for epoch in range(0, args.epochs):

        # Train model for one epoch
        train(train_loader, model, preprocess, ce_loss, optimizer, scheduler, classnames, labels, epoch)

        # Evaluate model on testing dataset
        accuracy = validate(val_loader, model, preprocess, ce_loss, classnames, labels, epoch)

        # Save the checkpoint
        if (epoch + 1) % args.save_epoch == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model,
                'accuracy': accuracy
            }, epoch + 1)


def train(train_loader, model, preprocess, ce_loss, optimizer, scheduler, classnames, labels, epoch):
    batch_time = AverageMeter()
    batch_loss = AverageMeter()
    batch_accuracy = AverageMeter()

    # Switch model to train mode
    model.train()

    end = time.time()
    num_batches = len(train_loader)
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Update learning rate with scheduler
        scheduler(i + epoch*num_batches)

        # Forward through CLIP vision and text encoder
        image_features, text_features, logit_scale = model(inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        # Calculate the loss function for CLIP
        loss = F.cross_entropy(logits, targets)
        
        # Backpropagate gradients and update model
        optimizer.zero_grad()
        model.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(mr_text, 1.0)
        optimizer.step()

        # Record the loss and elapsed time
        accuracy = get_clip_accuracy(logits.detach().clone(), targets)[0]
        batch_accuracy.update(accuracy, inputs.size(0))
        batch_loss.update(loss.data, inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Current Learning Rates: {lr}'.format(lr=get_lr(optimizer)))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time, 
                      loss=batch_loss, accuracy=batch_accuracy))
    
    print('---------------> Training Loss {loss.avg:.3f} <---------------'.format(loss=batch_loss))


def validate(val_loader, model, preprocess, ce_loss, classnames, labels, epoch):
    batch_time = AverageMeter()
    batch_loss = AverageMeter()
    batch_accuracy = AverageMeter()

    # Switch model to evaluation mode
    model.eval()

    end = time.time()
    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Forward through CLIP vision and text encoder
        with torch.no_grad():
            image_features, text_features, logit_scale = model(inputs)

        # Calculate the logits for CLIP
        logit_scale = logit_scale.mean()
        similarity = logit_scale * image_features @ text_features.t()

        # Calculate the loss function for CLIP
        loss = ce_loss(similarity, targets)

        # Measure top1 accuracy and record the loss
        accuracy = get_clip_accuracy(similarity.detach().clone(), targets)[0]
        batch_loss.update(loss.data, inputs.size(0))
        batch_accuracy.update(accuracy, inputs.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                      epoch, i, len(val_loader), batch_time=batch_time, 
                      loss=batch_loss, accuracy=batch_accuracy))

    print('---------------> Evaluation Accuracy {accuracy.avg:.3f} <---------------'.format(accuracy=batch_accuracy))
    return batch_accuracy.avg


def convert_image_to_rgb(image):
    return image.convert("RGB")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_clip_accuracy(similarity, targets, topk=(1,)):
    similarity = similarity.softmax(dim=-1)
    batch_size = targets.size(0)
    maxk = max(topk)

    _, pred = similarity.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    
    return res


def save_checkpoint(state, epoch):
    directory = "/nobackup/ageng/checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filename = directory + 'checkpoint_{epoch}.pth'.format(epoch=epoch)
    torch.save(state, filename)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()