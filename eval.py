import os
import yaml
import json
import torch
import random
import argparse
import numpy as np
from torch import nn
from torch.utils import data
from models import get_model
from loader import get_loader
from utils import  AverageMeter, accuracy, Confidence_Diagram, brier_score
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import log_loss
from optimizers import get_optimizer
from loss import get_loss_function


def eval(cfg, logdir):

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = cfg['data']['num_classes']

    # Setup Model
    model_cfg = {}
    for item in cfg["model"]:
        if cfg["model"][item] is not None:
            model_cfg[item] = cfg["model"][item]
    model_cfg['calibration'] = cfg['testing']['calibration']

    model = get_model(**model_cfg, num_classes=n_classes, device=device).to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.eval()

    # Resume pre-trained model
    if cfg["testing"]["resume"] is not None:
        if os.path.isfile(cfg["testing"]["resume"]):
            checkpoint = torch.load(cfg["testing"]["resume"])
            pretrained_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)
            
        else:
            print("No checkpoint found at '{}'".format(cfg["testing"]["resume"]))
                    

    #================================ AutoTune NLL ==========================================
    # Calculate the exponential mapping
    if cfg['testing']['exponential_map']:
        print('Calculating mean and std')
        v_loader = get_loader(cfg,'val')
        valloader = data.DataLoader(v_loader,
            batch_size=cfg["testing"]["batch_size"],num_workers=cfg["testing"]["n_workers"])
        with torch.no_grad():
            norms = None
            for image, _ in valloader:
                x_norms = model(image.to(device), is_feature_norm = True)
                if norms == None:
                    norms = x_norms
                else:
                    norms = torch.cat((x_norms, norms), dim=0)
            x_mu  = torch.mean(norms)
            x_std = torch.std(norms)
            c = -np.log(cfg['testing']['exponential_map'])/(x_mu-x_std).item()
    else:
        c = None
    
    # Tuning calibration parameters on the validation set
    if cfg['testing']['calibration'] != 'none': 
        print('============================== start auto-tuning ==============================================')
        # Initilize data loader and setup optimizer
        v_loader = get_loader(cfg,'val')
        valloader = data.DataLoader(
            v_loader, 
            batch_size=cfg["testing"]["batch_size"], 
            num_workers=cfg["testing"]["n_workers"]
        )
        optimizer_cls = get_optimizer(cfg["testing"]["optimizer"])
        optimizer_params = {k: v for k, v in cfg["testing"]["optimizer"].items() if k != "name"}

        # Set optimizable parameters
        if cfg['testing']['calibration'] == 'alpha-beta':
            optimizer = optimizer_cls(nn.ModuleList([model.module.decomp_a,model.module.decomp_b]).parameters(), **optimizer_params)
        else:
            optimizer = optimizer_cls(model.module.temp.parameters(), **optimizer_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['testing']['tune_epoch'])

        loss_fn = get_loss_function("CrossEntropy")

        for epoch in range(cfg['testing']['tune_epoch']):
            for i,(image, target) in enumerate(valloader):
                image = image.to(device)
                target = target.to(device)
                logit,_,_,_ = model(image, grad=False)
                loss = loss_fn(logit, target)
                # ODIR regularisation https://arxiv.org/pdf/1910.12656.pdf
                if cfg['testing']['calibration'] == 'matrix' or cfg['testing']['calibration'] == 'dirichlet':
                        weight_reg = 1e-7 * torch.norm(model.module.temp.weight * (1 - torch.eye(n_classes).to(device)))/(n_classes *(n_classes-1))
                        bais_reg =  1e-7 * torch.norm(model.module.temp.bias)/n_classes
                        loss = loss + weight_reg + bais_reg
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            print("Auto Tune (NLL) epoch:{}, lr:{lr:.5f},  loss:{loss:.3f}".format(epoch,lr=optimizer.param_groups[-1]['lr'], loss=loss.item()))
        print('============================== end auto-tuning ==============================================')
  #================================ Evaluation ==========================================
    print('============================== start evaluation ==============================================')
    # Specifying CIFAR100 groups 
    if cfg['testing']['dataset'] == 'cifar100' and cfg['testing']['group'] is not None:
        group_list = pd.read_csv('./datasets/cifar100_splits.csv')
        head = group_list.columns[1:][cfg['testing']['group']][0]
        group = group_list[head].values.tolist()[:-2]
    else:
        group = None

    # Ground truth and prediction lists
    gt_list = []
    pred_list = []

    # Calibration metrics: ECE, NLL, Brier
    ece_list = []
    nll_list = []
    brier_list = []
    
    # OOD detection metrics
    pred_prob_list = []
    norm_list = []
    sim_diff_list = []
    u_list = []
    energy_list = []

    if 'degredation' in cfg['testing'] and cfg['testing']['degredation']['type'] is not None:
        types = cfg['testing']['degredation']['type']
        noise_levels = cfg['testing']['degredation']['value']
        mode = 'test'
    else:
        types = [None]
        noise_levels = [None]
        mode = 'test'

    with torch.no_grad():
        for noise_type in types:
            for level in noise_levels:
                # Initilize accuracy and calibration metrics for each condition
                eval_top1 = AverageMeter('Acc@1', ':6.2f')
                eval_top5 = AverageMeter('Acc@5', ':6.2f')
                nll_avg_meter = AverageMeter('negative_log_likelihood', ':6.2f')
                brier_avg_meter = AverageMeter('mutual_info', ':6.2f')
                ece_meter = Confidence_Diagram(n_classes)

                # Update validation config file for each condition
                v_cfg = cfg
                if noise_type is not None:
                    v_cfg['testing']['degredation']['type'] = noise_type
                    v_cfg['testing']['degredation']['value'] = level
                v_loader = get_loader(v_cfg, mode, group=group)
                valloader = data.DataLoader(
                    v_loader, 
                    batch_size=cfg["testing"]["batch_size"], 
                    num_workers=cfg["testing"]["n_workers"]
                )

                # Start looping through valloader
                for i, (image, target) in enumerate(valloader):
                    image = image.to(device)
                    target = target.to(device)
                    batch_sz = image.shape[0]
                    logit, norm, sim, h_logit = model(image, c = c)
                    pred_dist = torch.nn.functional.softmax(logit,dim=1)
                    
                    # Save OOD scores in lists
                    sim_diff = sim.max(1)[0] - sim.mean(1)
                    sim_diff_list.extend(sim_diff.cpu().numpy())
                    norm_list.extend(norm.squeeze().cpu().numpy())
                    u_list.extend((sim_diff*norm.squeeze()).cpu().numpy())
                    pred_prob_list.extend(pred_dist.max(1)[0].cpu().numpy())
                    energy = torch.logsumexp(h_logit, dim=1, keepdim=False) 
                    energy_list.extend(energy.cpu().numpy())

                    # Save ground truth and predited class
                    gt_list.extend(target.cpu().numpy())
                    pred_list.extend(pred_dist.argmax(1).cpu().numpy())
                       
                    # Save accuracy
                    acc1, acc5 = accuracy(pred_dist, target, topk=(1, 5))
                    eval_top1.update(acc1[0], batch_sz)
                    eval_top5.update(acc5[0], batch_sz)

                    # Calculate and save expected calibration error
                    ece_meter.aggregate_stats(pred_dist,target)
                    
                    
                    # Calculate and save negative log likelihood 
                    nll_avg_meter.update(log_loss(target.cpu(),pred_dist.cpu(), labels = list(range(n_classes))))
                    
                    
                    # Calculate and save brier score
                    brier_avg_meter.update(brier_score(target,pred_dist).mean().cpu())
                    
                    
                    if i % 100 == 0:
                        output = ('Test:  [{0}/{1}]\t'
                                'Prec@1 {top1.avg:.3f}\t'
                                'Prec@5 {top5.avg:.3f}'.format(i, len(valloader), top1=eval_top1, top5=eval_top5))  
                        print(output)
                
                # Append the averaged ECE, NLL, and Brier for this noise and the corresponding serverity level
                ece_meter.compute_ece()
                ece_list.append(ece_meter.ece)
                nll_list.append(nll_avg_meter.avg.item())
                brier_list.append(brier_avg_meter.avg.item())
        
        
    calibration_summary = ('Test: \t' 'Prec@1 {top1.avg:.3f}\t' 'ECE {ece:.4f}\t'  'NLL {nll:.4f}\t'  'Brier {brier:.4f}\t'
    .format(top1=eval_top1, ece=np.mean(ece_list), nll=np.mean(nll_list), brier=np.mean(brier_list)))
    print(calibration_summary)
    
    # Write calibration results to txt file
    dir = os.path.join(logdir,cfg['testing']['calibration'])
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(os.path.join(dir, cfg['testing']['dataset'] + '_calibration.txt'),'w') as file:
        file.write(calibration_summary)

    # Write scores to csv files for OOD detection in OOD.py
    ood_score_summary = {'gt':gt_list,'pred':pred_list, 'prob':pred_prob_list, 'norm':norm_list , 'sim_diff':sim_diff_list, 'u_list':u_list , 'energy':energy_list} 
    df = pd.DataFrame(ood_score_summary)
    score_filename = os.path.join(dir, cfg['testing']['dataset'] + '_scores')  #logdir + '/' + cfg['testing']['calibration'] +'/ood_' + cfg['testing']['dataset'] + '_scores'
    if cfg['testing']['dataset'] == 'cifar100' and cfg['testing']['group'] is not None:
        score_filename = score_filename + '_' + str(cfg['testing']['group'][0])

    
    df.to_csv(score_filename + '.csv')
    f = open(score_filename + '.yaml', "w")
    j = json.dumps(cfg, indent=4)
    f.write(j)
    f.close()
    print('Done')

        
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
        cfg = yaml.load(fp)
    logdir = os.path.join("runs", 'test', cfg["data"]["name"], cfg["model"]["arch"], cfg['id'])
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    print("RUNDIR: {}".format(logdir))
    eval(cfg, logdir)