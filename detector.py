import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from metrics import metrics
from utils import get_dataset
# from torchvision.utils import save_image,make_grid
import matplotlib.pyplot as plt
import abc

class Detector(metaclass=abc.ABCMeta):
    def calculate_metrics(self):
        self.auroc,_ = self.met.AUROC_AUPR()
        _,self.fpr95tpr,_ = self.met.FPR_at_95TPR()
        return self.auroc,self.fpr95tpr

    def plot_AUROC(self,fig1=None,fig2=None,name=['ROC','PR']):
        self.met.plot_ROC_PR(fig1=fig1,fig2=fig2,name=name)

    def plot_score_density(self,fig = None,legend = True):
        if fig is None:
            plt.figure(figsize=(8,8))
            fig = plt.gca()
        fig.hist(self.pred[:self.pos_num],bins=200,label='In-Distribution')
        fig.hist(self.pred[self.pos_num:],bins=200,alpha=0.5,label='Out-of-Distribution')
        if legend:
            fig.legend(fontsize=22)


class Baseline_Detector(Detector):
    def __init__(self,model,indataloader,outdataloader,device=torch.device('cpu')):
        self.model = model.to(device)
        self.device = device
        self.label = []
        self.pred = []

        # Positive
        for (data,_) in indataloader:
            self.pred+=self.eval(data)
        self.pos_num = len(self.pred)
        self.label += [1]*self.pos_num

        # Negative
        for (data,_) in outdataloader:
            self.pred+=self.eval(data)
        self.neg_num = len(self.pred)-self.pos_num
        self.label += [0]*self.neg_num

        self.met = metrics(self.pred,self.label)

        

    def eval(self,data):
        """
        Input: images
        Output: softmax scores
        """
        self.model.eval()
        data = data.to(self.device)
        out = self.model(data)
        out,_ = F.softmax(out,dim=1).max(dim=1)
        return out.clone().cpu().detach().numpy().tolist()

class ODIN_Detector(Detector):
    def __init__(self,model,indataloader,outdataloader,
                epsilon=0.0014,T=1000,IPP=True,
                device=torch.device('cpu')):
        self.model = model.to(device)
        self.device = device
        self.epsilon = epsilon
        self.T = T
        self.label = []
        self.pred = []

        # Positive
        for (data,_) in indataloader:
            if IPP:
                data = self.input_preprocessing(data)
            self.pred+=self.eval(data)
        self.pos_num = len(self.pred)
        self.label += [1]*self.pos_num

        # Negative
        for (data,_) in outdataloader:
            if IPP:
                data = self.input_preprocessing(data)
            self.pred+=self.eval(data)
        self.neg_num = len(self.pred)-self.pos_num
        self.label += [0]*self.neg_num

        self.met = metrics(self.pred,self.label)

        

    def eval(self,data):
        """
        Input: images
        Output: softmax scores
        """
        self.model.eval()
        data = data.to(self.device)
        out = self.model(data)
        out = out/self.T
        out,_ = F.softmax(out,dim=1).max(dim=1)
        return out.clone().cpu().detach().numpy().tolist()

    def input_preprocessing(self,data):
        dat = data.clone().detach().to(self.device)
        dat.requires_grad_(True)
        self.model.eval()
        out = self.model(dat)
        out = out/self.T
        _,label = out.max(dim=1)
        loss = F.cross_entropy(out,label.detach())
        loss.backward()
        norm = torch.tensor((63.0/255, 62.1/255.0, 66.7/255.0)).reshape([1,3,1,1]).to(self.device)
        dat = dat-self.epsilon*torch.sign(dat.grad.data)/norm
        return dat.detach()


if __name__ == '__main__':
    # nn.Module.dump_patches=True
    indataloader=get_dataset('cifar10',16,4)
    outdataloader = get_dataset('LSUN',16,4)
    net = torch.load("./models/densenet10.pth")
    # Det = ODIN_Detector(net,indataloader,outdataloader,T=1000,epsilon=0.0014,device = torch.device('cuda'))
    Det = Baseline_Detector(net,indataloader,outdataloader,device=torch.device('cuda'))
    auroc,fpr = Det.calculate_metrics()
    Det.plot_AUROC()
    Det.plot_score_density()
    plt.show()
    print(auroc,fpr)
