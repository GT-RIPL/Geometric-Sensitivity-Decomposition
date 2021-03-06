import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils import data
from models import get_model
from loader import get_loader
from optimizers import get_optimizer
from loss import get_loss_function
from utils import save_checkpoint, AverageMeter, accuracy, get_logger

def train(cfg, writer, logger, logdir):

    # Setup seeds
    if cfg['training']['seed'] is not None:
        seed = cfg['training']['seed']
        logger.info("Using seed {}".format(seed))
        torch.manual_seed(cfg.get("seed", cfg['training']['seed']))
        torch.cuda.manual_seed(cfg.get("seed", cfg['training']['seed']))
        np.random.seed(cfg.get("seed", cfg['training']['seed']))
        random.seed(cfg.get("seed", cfg['training']['seed']))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    t_loader = get_loader(cfg,'train')
    v_loader = get_loader(cfg,'val')

    trainloader = data.DataLoader(
        t_loader,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
    )

    valloader = data.DataLoader(
        v_loader, 
        batch_size=cfg["training"]["batch_size"], 
        num_workers=cfg["training"]["n_workers"]
    )
    n_classes = cfg['data']['num_classes']
    
    
    # Setup Model 
    model_cfg = {}
    for item in cfg["model"]:
        if cfg["model"][item] is not None:
            model_cfg[item] = cfg["model"][item]
    model = get_model(**model_cfg, num_classes=n_classes, device=device).to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg["training"]["optimizer"])
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))
    loss_fn = get_loss_function(cfg["training"]["loss"])
    logger.info("Using loss {}".format(loss_fn))

    if cfg['training']['scheduler'] == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['train_epoch'])
    elif cfg['training']['scheduler'] == 'wide':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 120, 160], 0.2)

    start_epoch = 0
    best_acc1 = -100.0
    
    # Resume pre-trained model
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"])
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            best_acc1 = checkpoint["best_acc1"]
            start_epoch = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

  #================================ Training ==========================================
    for epoch in range(start_epoch, cfg['training']['train_epoch']):
       
        # Initilize trianing statistics 
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        
        end = time.time()
        model.train()  
        for i,(image, target) in enumerate(trainloader):
            data_time.update(time.time() - end)
            image = image.to(device)
            target = target.to(device)
            logit,_,_,_= model(image) 
            loss = loss_fn(logit, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Measure accuracy and record loss
            acc1, acc5 = accuracy(logit, target, topk=(1, 5))
            losses.update(loss.item(), image.size(0))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg["training"]["print_interval"] == 0:
                output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(trainloader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))  # TODO
                print(output)
                logger.info(output + '\n')
    
        # Write stats to tensorboard
        writer.add_scalar('train/loss', losses.avg, epoch)
        writer.add_scalar('train/acc_top1', top1.avg, epoch)
        writer.add_scalar('train/acc_top5', top5.avg, epoch)
        writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

        # Update scheduler for learning rate decay
        scheduler.step()
        ##================== Evaluation ============================
        
        model.eval()
        with torch.no_grad():
            eval_top1 = AverageMeter('Acc@1', ':6.2f')
            eval_top5 = AverageMeter('Acc@5', ':6.2f')
            val_losses = AverageMeter('Loss', ':.4e')
            

            for i, (image, target) in enumerate(valloader):
                image = image.to(device)
                target = target.to(device)
                logit,_,_,_ = model(image) 
                loss = loss_fn(logit, target)

                acc1, acc5 = accuracy(logit, target, topk=(1, 5))
                eval_top1.update(acc1[0], image.size(0))
                eval_top5.update(acc5[0], image.size(0))
                val_losses.update(loss.item(), image.size(0))

                if i % cfg["training"]["print_interval"] == 0:
                    output = ('Test: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                            'Prec@1 {top1.avg:.3f}\t'
                            'Prec@5 {top5.avg:.3f}'.format(
                        epoch, i, len(valloader), top1=eval_top1, top5=eval_top5, lr=optimizer.param_groups[-1]['lr']))  # TODO
                    print(output)

                    output = ('validation Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                            .format(top1=eval_top1, top5=eval_top5))
                    logger.info(output + '\n')

                writer.add_scalar('/loss', val_losses.avg, epoch)
                writer.add_scalar('/acc_top1', eval_top1.avg, epoch)
                writer.add_scalar('/acc_top5', eval_top5.avg, epoch)

            is_best = eval_top1.avg > best_acc1
            best_acc1 = max(eval_top1.avg, best_acc1)
       
        output_best = 'Best Prec@1: %.3f' % (best_acc1)
        logger.info(output_best + '\n')
        print(output_best)
        
        if (epoch+1) %200 == 0:
            filename = '{}/ckpt.{}.pth.tar'.format(logdir,epoch)
            torch.save({
                'epoch': epoch,
                'arch': cfg["model"]["arch"],
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, filename)

        save_checkpoint(logdir, {
            'epoch': epoch,
            'arch': cfg["model"]["arch"],
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="./configs/resnet18/wide.yaml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)
        logdir = os.path.join("runs",cfg["data"]["name"],cfg["model"]["arch"],cfg['id'])
        if not os.path.exists(logdir):
            os.makedirs(logdir, exist_ok=True)
        print("RUNDIR: {}".format(logdir))
        shutil.copy(args.config, logdir)
        writer = SummaryWriter(logdir)
        logger = get_logger(logdir)
        logger.info("Let the games begin")
        train(cfg, writer, logger, logdir)