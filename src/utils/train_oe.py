import time
import torch
import numpy as np
from utils import step_lr, cosine_lr


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


class OELoss(torch.nn.Module):
    def __init__(self):
        super(OELoss, self).__init__()

    def forward(self, id_inputs, ood_inputs):
        return -((ood_inputs.mean(1) - torch.logsumexp(ood_inputs, dim=1)).mean() - np.log(1/ood_inputs.shape[1]))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_accuracy(similarity, targets, topk=(1,)):
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


def train_oe(train_loader, ood_loader, model, id_criterion, ood_criterion, optimizer, scheduler, epoch, label_map, beta=0.5, print_freq=75):
    batch_time = AverageMeter()
    batch_id_loss = AverageMeter()
    batch_ood_loss = AverageMeter()
    batch_accuracy = AverageMeter()

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    ood_loader.dataset.offset = np.random.randint(len(ood_loader.dataset))

    # Switch model to train mode
    model.train()

    end = time.time()
    num_batches = len(train_loader)
    for i, (id_samples, ood_samples) in enumerate(zip(train_loader, ood_loader)):
        inputs = torch.cat((id_samples[0], ood_samples[0]), 0).cuda()
        targets = id_samples[1].cuda()

        # Update learning rate with scheduler
        scheduler(i + epoch*num_batches)

        # Mapping true labels to subset labels
        targets = torch.Tensor([label_map[target.item()] for target in targets]).long().cuda()

        # Forward through CLIP vision encoder
        outputs = model(inputs)

        # Calculate the In-distribution loss for model
        id_output = outputs[:len(id_samples[0])]
        id_loss = id_criterion(id_output, targets)

        # Calculate the Out-of-distribution loss for model
        ood_output = outputs[len(id_samples[0]):]
        ood_loss = ood_criterion(id_output, ood_output)

        # Calculate the full loss for the model
        loss = id_loss + beta*ood_loss
        
        # Backpropagate gradients and update model
        optimizer.zero_grad()
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure top1 accuracy and record the loss
        accuracy = get_accuracy(id_output.detach().clone(), targets)[0]
        batch_id_loss.update(id_loss.data, len(id_samples[0]))
        batch_ood_loss.update(ood_loss.data, len(ood_samples[0]))
        batch_accuracy.update(accuracy, inputs.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Current Learning Rate: {lr}'.format(lr=get_lr(optimizer)))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'ID Loss {id_loss.val:.4f} ({id_loss.avg:.4f})\t'
                  'OOD Loss {ood_loss.val:.4f} ({ood_loss.avg:.4f})\t'
                  'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      id_loss=batch_id_loss, ood_loss=batch_ood_loss, accuracy=batch_accuracy))
    
    print('---------------> Training Accuracy {accuracy.avg:.3f} <---------------'.format(accuracy=batch_accuracy))


def validate(test_loader, model, image_loss, epoch, label_map, print_freq=75):
    batch_time = AverageMeter()
    batch_loss = AverageMeter()
    batch_accuracy = AverageMeter()

    # Switch model to evaluation mode
    model.eval()

    end = time.time()
    for i, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Mapping true labels to subset labels
        targets = torch.Tensor([label_map[target.item()] for target in targets]).long().cuda()

        # Forward through CLIP vision encoder
        with torch.no_grad():
            outputs = model(inputs)

        # Calculate the loss function for CLIP
        loss = image_loss(outputs, targets)

        # Measure top1 accuracy and record the loss
        accuracy = get_accuracy(outputs.detach().clone(), targets)[0]
        batch_loss.update(loss.data, inputs.size(0))
        batch_accuracy.update(accuracy, inputs.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if print_freq is not None and i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                      epoch, i, len(test_loader), batch_time=batch_time, 
                      loss=batch_loss, accuracy=batch_accuracy))

    print('---------------> Evaluation Accuracy {accuracy.avg:.3f} <---------------'.format(accuracy=batch_accuracy))
    return batch_accuracy.avg


def train_oe_model(train_loader, ood_loader, test_loader, model, learning_rate, weight_decay, warmup_length, epochs, label_map):

    # Defining loss function and optimizer
    id_criterion = torch.nn.CrossEntropyLoss().cuda()
    ood_criterion = OELoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    
    # Defining learning rate scheduler
    scheduler = cosine_lr(optimizer, learning_rate, warmup_length, epochs*len(train_loader))

    # Iterating through training epochs
    for epoch in range(0, epochs):

        # Train model for one epoch
        train_oe(train_loader, ood_loader, model, id_criterion, ood_criterion, optimizer, scheduler, epoch, label_map, beta=0.5)

        # Evaluate model on testing dataset
        id_accuracy = validate(test_loader, model, id_criterion, epoch, label_map)

    return model, id_accuracy