from __future__ import print_function

import argparse
from argparse import ArgumentParser

import os
import datetime
import sys
import time

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import torch_optimizer as optim

torch.manual_seed(0)

from utils import get_git_diff, get_git_hash, Logger
from models import preact_resnet110_cifar
from torch import nn

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler
from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler

from ignite.contrib.engines import common

from ignite.contrib.handlers.param_scheduler import PiecewiseLinear


try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        raise RuntimeError(
            "This module requires either tensorboardX or torch >= 1.2.0. "
            "You may install tensorboardX with command: \n pip install tensorboardX \n"
            "or upgrade PyTorch using your package manager of choice (pip or conda)."
        )
class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

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


def loss_fn(args, output, target, model, device, reduction='mean'):
  if args.wd > 0:
    reg_loss = 0.0
    for name, weight in model.named_parameters():
      if 'weight' in name:
        reg_loss += torch.sum(weight ** 2)
    reg_loss *= 0.5 * args.wd
  else:
    reg_loss = 0.0
  if args.bd > 0:
    regb_loss = 0.0
    for name, bias in model.named_parameters():
      if 'bias' in name:
        regb_loss += torch.sum(bias ** 2)
    regb_loss *= 0.5 * args.bd
    reg_loss += regb_loss
  if args.loss == 'nll':
    loss = F.nll_loss(output, target, reduction=reduction) + reg_loss
  else:
    onehot_target = torch.zeros((target.shape[0], 10),
                                dtype=torch.float32, device=device)
    onehot_target.scatter_(1, target.view(-1, 1), 1)
    loss = F.mse_loss(torch.exp(output), onehot_target,
                      reduction=reduction) + reg_loss
  return loss


def get_data_loaders(batch_size):
  train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
      root=args.data_dir, train=True, download=True,
      transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
          (0.247, 0.243, 0.262)),
        ])), batch_size=batch_size, shuffle=True, num_workers=2,)

  test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
      root=args.data_dir, train=False, download=True,
      transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.262)),
      ])), batch_size=batch_size, shuffle=False, num_workers=2,
    )
  return train_loader, test_loader

def run_exp(args):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  if device == 'cuda':
    cudnn.benchmark = True
  print("Using: {}".format(device))
  
  batch_size = args.bs
  train_loader, val_loader = get_data_loaders(batch_size)
  writer = SummaryWriter(log_dir=args.output_dir)

  model = preact_resnet110_cifar().to(device)
  
  if args.nu_mode == 'NAG':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
  elif args.nu_mode == 'QHM':
    optimizer = optim.QHM(model.parameters(), lr=args.lr, momentum=args.momentum, nu=args.nu)
  elif args.nu_mode == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  elif args.nu_mode == 'RMSProp':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
  

  criterion = nn.NLLLoss()

  trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

  val_metrics = {"accuracy": Accuracy(), "nll": Loss(criterion)}
  evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)


  if args.checkpoint:
    if os.path.isfile(args.checkpoint):
      print('=> loading checkpoint "{}"'.format(args.checkpoint))
      checkpoint = torch.load(args.checkpoint)
      model.load_state_dict(checkpoint['state_dict'])
    else:
      print("=> no checkpoint found at '{}'".format(args.checkpoint))

  if args.lr_drop_mode == 'freq':
    step_scheduler = torch.optim.lr_scheduler.StepLR(
      optimizer, args.drop_freq, args.drop_rate,
    )
    lr_scheduler = LRScheduler(step_scheduler)
  elif args.lr_drop_mode == 'steps':
    step_scheduler = torch.optim.lr_scheduler.MultiStepLR(
      optimizer, args.drop_steps, args.drop_rate,
    )
    lr_scheduler = LRScheduler(step_scheduler)
  elif args.lr_drop_mode == 'onecycle':
    
    lr_scheduler = LinearCyclicalScheduler(optimizer, 'lr', args.lr, 1e-3, len(train_loader))
  elif args.lr_drop_mode == 'cosineannealing':
    
    lr_scheduler = CosineAnnealingScheduler(optimizer, 'lr', args.lr, 1e-3, len(train_loader))

  else:
    raise ValueError('Unknown drop_mode: {}'.format(args.drop_mode))
    

  if args.momentum_drop_mode == 'yes':
    constant = args.momentum/(1-args.momentum)
    momentum_scheduler = PiecewiseLinear(optimizer, "momentum", milestones_values=[(5, args.momentum), (6, 2*constant/(2*constant+1)), (15, 2*constant/(2*constant+1)), (16, 4*constant/(4*constant+1)), (35, 4*constant/(4*constant+1)), (36, 8*constant/(8*constant+1)), (75, 8*constant/(8*constant+1))])
  else:
    momentum_scheduler = PiecewiseLinear(optimizer, "momentum", milestones_values=[(5, args.momentum), (15, args.momentum), (35, args.momentum), (75, args.momentum)])

  if args.bs_drop_mode == 'yes':
    bs_scheduler = PiecewiseLinear(optimizer, "batch_size", milestones_values=[(5, args.bs), (6, args.bs/2), (15, args.bs/2), (16, args.bs/4), (35, args.bs/4), (36, args.bs/8), (75, args.bs/8)])
  else:
    bs_scheduler = PiecewiseLinear(optimizer, "batch_size", milestones_values=[(5, args.bs), (15, args.bs), (35, args.bs), (75, args.bs)])

  
    
  step = 0
  train_writer = SummaryWriter(os.path.join(args.output_dir, 'logs', 'train'))
  test_writer = SummaryWriter(os.path.join(args.output_dir, 'logs', 'test'))


  @trainer.on(Events.ITERATION_COMPLETED(every=args.log_interval))
  def log_training_loss(engine):
    print(
      f"Epoch[{engine.state.epoch}] Iteration[{engine.state.iteration}/{len(train_loader)}]" 
      f"Loss: {engine.state.output:.2f}")
    writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)
  
  @trainer.on(Events.EPOCH_COMPLETED)
  def log_training_results(engine):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics["accuracy"]
    avg_nll = metrics["nll"]
    print(f"Training Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.3f} Avg loss: {avg_nll:.3f}")
    writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)
    writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.epoch)

  @trainer.on(Events.EPOCH_COMPLETED)
  def log_validation_results(engine):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics["accuracy"]
    avg_nll = metrics["nll"]
    print(f"Validation Results - Epoch: {engine.state.epoch} Avg accuracy: {avg_accuracy:.3f} Avg loss: {avg_nll:.3f}")
    writer.add_scalar("validation/avg_loss", avg_nll, engine.state.epoch)
    writer.add_scalar("validation/avg_accuracy", avg_accuracy, engine.state.epoch)


  if args.lr_drop_mode == 'cosineannealing':
   trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)
  elif args.lr_drop_mode == 'onecycle':
   trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)
  else:
    trainer.add_event_handler(Events.EPOCH_STARTED, lr_scheduler)
  
  
  trainer.add_event_handler(Events.EPOCH_STARTED, bs_scheduler)
  trainer.add_event_handler(Events.EPOCH_STARTED, momentum_scheduler)
  trainer.run(train_loader, max_epochs=args.epoch)

  writer.close()
  
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bs", type=int, default=256, help="input batch size for training (default: 256)")
    parser.add_argument("--epoch", type=int, default=75, help="number of epochs to train (default: 75)")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate (default: 0.1)")
    parser.add_argument("--nu", type=float, default=0.5, help="nu (default: 0.5)")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)")
    parser.add_argument("--log_interval", type=int, default=10, help="how many batches to wait before logging training status")
    parser.add_argument("--output_dir", type=str, default="tensorboard_logs", help="log directory for Tensorboard log output")

    
    parser.add_argument('--data_dir', dest='data_dir', type=str,
                      default=os.getenv('PT_DATA_DIR', 'data'))
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10, help="In epochs")

    parser.add_argument('--drop_freq', type=int, default=25)
    parser.add_argument('--drop_steps', nargs='+', type=int,
                      default=[5, 15, 35, 75])
    parser.add_argument('--drop_rate', type=float, default=0.5)
    
    
    
    parser.add_argument('--bs_drop_mode', choices=['yes', 'no'], default='no')
    
    parser.add_argument('--lr_drop_mode', choices=['steps', 'freq', 'onecycle', 'cosineannealing'], default='steps')
    
    parser.add_argument('--momentum_drop_mode', choices=['yes', 'no'], default='no')
    parser.add_argument('--nu_mode', choices=['NAG', 'QHM', 'RMSProp', 'Adam'], default='QHM')
    
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--bd', type=float, default=0.0, help='bias decay')
    parser.add_argument('--checkpoint', type=str, default="", 
                      help="Checkpoint to restore the model from")

    args = parser.parse_args()

    
    
    logdir = args.output_dir
    os.mkdir(logdir)
    os.makedirs(os.path.join(logdir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(logdir, 'models'), exist_ok=True)
    tm_suf = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_log = open(
      os.path.join(logdir, 'stdout_{}.log'.format(tm_suf)), 'a', 1,
    )
    stderr_log = open(
      os.path.join(logdir, 'stderr_{}.log'.format(tm_suf)), 'a', 1,
    )
    sys.stdout = Logger(sys.stdout, stdout_log)
    sys.stderr = Logger(sys.stderr, stderr_log)

    run_exp(args)

    sys.stdout = old_stdout
    sys.stderr = old_stderr
    stdout_log.close()
    stderr_log.close()
