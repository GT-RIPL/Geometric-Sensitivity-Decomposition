import torch
import shutil
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import logging
import datetime 

def calc_confusion_mat(val_loader, model, args):
    
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    cf = confusion_matrix(all_targets, all_preds).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt

    print('Class Accuracy : ')
    print(cls_acc)
    classes = [str(x) for x in args.cls_num_list]
    plot_confusion_matrix(all_targets, all_preds, classes)
    plt.savefig(os.path.join(args.root_log, args.store_name, 'confusion_matrix.png'))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def prepare_folders(args):
    
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def prepare_folders_eval(args):
    
    folders_util = [args.root_eval, 
                    os.path.join(args.root_eval, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def save_checkpoint(logdir, state, is_best):
    
    filename = '%s/ckpt.pth.tar' % logdir
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class VarianceRatioMeter(object):
    
    def __init__(self, name, num_classes:int, batch_sz: int, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.num_classes = num_classes
        self.batch_sz = batch_sz
        self.counts = torch.zeros((batch_sz, num_classes))
        self.reset()

    def reset(self):
        self.class_win_list = [0] * self.num_classes
        self.count = 0

    def update(self, logit, n=1):
        self.count += n
        val, index = torch.topk(logit, 1)
        index = index.squeeze(1)
        for i in range(self.batch_sz):
            idx = index[i]
            self.counts[i, idx] += 1
    
    def get_winner(self, wrong_idxes = None):
        counts = self.counts[wrong_idxes,:] if wrong_idxes is not None else self.counts
        val, _ = torch.topk(counts, 1)
        vr = 1 - (val.squeeze()/counts.sum(1))
        return torch.mean(vr)

def accuracy(output, target, topk=(1,)):
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Confidence_Diagram():
    def __init__(self,n_classes):
        # self.logdir = logdir
        self.num_bins = 15
        self.num_class = n_classes
        self.correct = np.zeros((self.num_bins))
        self.count = np.zeros((self.num_bins))
        self.confidence = np.zeros((self.num_bins))

    def aggregate_stats(self, dist, targets):
        # dist [batch, n_classes]
        # targets [batch,]
        
        prob, pred = dist.max(1)
        bins = prob // 0.1 #[batch,] 
        bins = bins.masked_fill(bins == self.num_bins,9)
        
        
        for i in np.linspace(0,9,self.num_bins):
            mask_bin = bins == i
            self.correct[int(i)] += (pred[mask_bin] == targets[mask_bin]).sum()
            self.count[int(i)] += (mask_bin).sum()
            self.confidence[int(i)] +=  prob[mask_bin].sum()
        

    def compute_ece(self):  
        self.accuracy = self.correct/np.maximum(self.count,1)
        self.confidence = self.confidence/np.maximum(self.count,1)
        ratio = self.count/np.expand_dims(np.nansum(self.count,0),0)
        self.ece = np.nansum(ratio*abs(self.accuracy - self.confidence),0)
        # self.ece = np.sum(self.ece_cls)/self.num_class
        
    def save(self,logdir):
        with open(os.path.join(logdir,'calibration.csv'), mode='w') as calibration:
            writer = csv.writer(calibration, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # import ipdb;ipdb.set_trace()
            writer.writerow(['mean accuracy per bin:']+(np.nansum(self.accuracy,1)/self.num_class).tolist())
            writer.writerow(['ece per class:'] + self.ece_cls.tolist())
            writer.writerow(['mean ece:',self.ece])

    def print(self):
        print(np.nansum(self.accuracy,1)/self.num_class)
        print(self.ece_cls)
        print(self.ece)

def get_logger(logdir, name = 'cnpt'):
    logger = logging.getLogger(name)
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

def brier_score(target, y_pred):
  pred_gt = y_pred.gather(1,target.unsqueeze(1)).squeeze()
  return (y_pred.square().sum(1) - 2 * pred_gt + 1)/y_pred.shape[1]

def fix_ckpt_dict(checkpoint):
    for key in list(checkpoint['state_dict'].keys()):
        if 'module' in key:
            new_key = key.replace('module.', '')
            checkpoint['state_dict'][new_key] = checkpoint['state_dict'][key]
            del checkpoint['state_dict'][key]
        else:
            break

# def brier_score(target, y_pred):
#     import ipdb;ipdb.set_trace()
#     nlabels = y_pred.shape[-1]
#     flat_probs = y_pred.reshape([-1, nlabels])
#     flat_labels = target.reshape([len(flat_probs)])

#     plabel = flat_probs[np.arange(len(flat_labels)), flat_labels]
#     out = np.square(flat_probs.cpu()).sum(axis=-1) - 2 * plabel.cpu()
#     import ipdb;ipdb.set_trace()
#     return out.reshape(labels.shape)