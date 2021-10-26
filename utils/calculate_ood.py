import numpy as np
from sklearn import metrics
import pandas as pd
import argparse
import os

def tnr_at_tpr95(ind_conf, ood_conf, tpr_threshold=0.95):
    sorted_ = np.sort(ind_conf)
    conf_threshold = sorted_[int((1-tpr_threshold)*(ind_conf.shape[0]))]

    ind_conf_ = np.zeros(ind_conf.shape, dtype=np.bool)
    ood_conf_ = np.zeros(ood_conf.shape, dtype=np.bool)
    ind_conf_[ind_conf > conf_threshold] = True
    ood_conf_[ood_conf > conf_threshold] = True

    ind = np.concatenate([ind_conf_, ood_conf_])
    ood = np.concatenate([np.ones(ind_conf.shape[0], dtype=np.bool), np.zeros(ood_conf.shape[0], dtype=np.bool)])

    tn = np.count_nonzero(~ind & ~ood)
    fp = np.count_nonzero(ind & ~ood)

    fpr = fp/(fp+tn)
    return 1-fpr

def AUROC(ind_conf, ood_conf):
    y_true = np.concatenate((np.ones(ood_conf.shape[0]),np.ones(ind_conf.shape[0])*2))
    y_score = np.concatenate((ood_conf,ind_conf))
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=2)
    return  metrics.auc(fpr, tpr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--train",
        type=str,
        default="cifar10",
        help="specify a training dataset used",
    )

    parser.add_argument(
        "--test",
        type=str,
        default="cifar10c",
        help="specify a testing dataset used",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet_vanilla",
        help="specify a model",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="none",
        help="specify a calibration used",
    )

    
    args = parser.parse_args()
    logdir = os.path.join('./runs/test/',args.train, 'wide_resnet28_10', args.model)
    ood_data_path = os.path.join(logdir, args.calibration, args.test+'_scores.csv')
    id_data_path = os.path.join(logdir, args.calibration, args.train+'_scores.csv')
    
    id_dataset = np.genfromtxt(id_data_path, delimiter=',',skip_header=1)
    ood_dataset = np.genfromtxt(ood_data_path, delimiter=',',skip_header=1)

    metric_list = ['index','gt','pred','prob','norm','sim_diff','u', 'energy']
    auroc_dict = {metric:[] for metric in metric_list[3:]}
    tpr95_dict = {metric:[] for metric in metric_list[3:]}


    for i in range(3,8):
        metric = i
        auc = AUROC(id_dataset[:,metric],ood_dataset[:,metric])
        tpr95 = tnr_at_tpr95(id_dataset[:,metric],ood_dataset[:,metric])
        auroc_dict[metric_list[i]].append(auc)
        tpr95_dict[metric_list[i]].append(tpr95)
    
    df = pd.DataFrame(auroc_dict).T
    df.to_csv(os.path.join(logdir, args.calibration, args.test+'_auroc_results.csv'))
    df = pd.DataFrame(tpr95_dict).T
    df.to_csv(os.path.join(logdir, args.calibration, args.test+'_tpr95_results.csv'))
