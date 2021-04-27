import numpy as np
import copy
import matplotlib.pyplot as plt

class metrics():
    def __init__(self,preds,labels):
        """
        param:
            preds: softmax predictions, e.g., [0.9,0.8,0.5,0.4,...]
            labels: positive/negative labels, e.g., [1,1,0,0,...]
        """
        self.preds = np.array(preds)
        self.labels = np.array(labels,dtype=np.int64)
        self.pos_index = np.where(self.labels==1)
        self.neg_index = np.where(self.labels==0)

    def calculate_metrics(self,th):
        """
        Calculate Recall, Precision, TPR, FPR with given threshold
        params:
            th: threshold
        return:
            recall
            precision
            tpr
            fpr
        """
        TP = np.sum(self.preds[self.pos_index]>=th)
        FP = np.sum(self.preds[self.neg_index]>=th)
        TN = np.sum(self.preds[self.neg_index]<th)
        FN = np.sum(self.preds[self.pos_index]<th)

        try:
            recall=TP/(TP+FN)
            precission=TP/(TP+FP)
            tpr=TP/(TP+FN)
            fpr=FP/(FP+TN)
        except ZeroDivisionError:
            recall,precission,tpr,fpr=0,0,0,0

        return recall,precission,tpr,fpr

    def list_metrics(self,th_list):
        """
        Calculate metrics with different th
        params:
            th_list: thresholds for evaluation
        return:
            rec_ls: recall list
            pre_ls: precision list
            tpr_ls: tpr list
            fpr_ls: fpr list
        """
        rec_ls = []
        pre_ls = []
        tpr_ls = []
        fpr_ls = []

        for th in th_list:
            rec,pre,tpr,fpr = self.calculate_metrics(th)
            rec_ls.append(rec)
            pre_ls.append(pre)
            tpr_ls.append(tpr)
            fpr_ls.append(fpr)

        rec_ls = np.array(rec_ls)
        pre_ls = np.array(pre_ls)
        tpr_ls = np.array(tpr_ls)
        fpr_ls = np.array(fpr_ls)

        roc_sort = np.argsort(fpr_ls)
        pr_sort = np.argsort(rec_ls)

        rec_ls = rec_ls[pr_sort]
        pre_ls = pre_ls[pr_sort]
        tpr_ls = tpr_ls[roc_sort]
        fpr_ls = fpr_ls[roc_sort]

        return rec_ls,pre_ls,tpr_ls,fpr_ls

    def AUROC_AUPR(self,th_list=None):
        if th_list is None:
            th_list=np.linspace(0,1.0,100001)
        rec_ls,pre_ls,tpr_ls,fpr_ls=self.list_metrics(th_list)
        auroc = 0.0
        aupr = 0.0
        for i in range(len(fpr_ls)-1):
            auroc += (tpr_ls[i]+tpr_ls[i+1])*(fpr_ls[i+1]-fpr_ls[i])
            aupr += (pre_ls[i]+pre_ls[i+1])*(rec_ls[i+1]-rec_ls[i])
        auroc *= 0.5
        aupr *= 0.5
        return auroc,aupr

    def plot_ROC_PR(self,th_list=None,fig1=None,fig2=None,name=['ROC','PR']):
        if th_list is None:
            th_list=np.linspace(0,1.0,100001)
        rec_ls,pre_ls,tpr_ls,fpr_ls=self.list_metrics(th_list)
        if fig1 is None:
            plt.figure(figsize=(8,8))
            fig1 = plt.gca()
        
        fig1.plot(fpr_ls,tpr_ls,label=name[0])
        # fig1.legend(fontsize=22)

        if fig2 is None:
            plt.figure(figsize=(8,8))
            fig2 = plt.gca()
        
        fig2.plot(rec_ls,pre_ls,label=name[1])
        # fig2.legend(fontsize=22)

    def FPR_at_95TPR(self,epsilon=1e-5,minth=0.0,maxth=1.0):
        th = (minth+maxth)/2
        _,_,tpr,fpr = self.calculate_metrics(th)
        while abs(tpr-0.95)>epsilon:
            if tpr>0.95:
                minth = th
            else:
                maxth = th
            th = (minth+maxth)/2
            _,_,tpr,fpr = self.calculate_metrics(th)
        # print(tpr,fpr,th)
        return tpr,fpr,th







        
