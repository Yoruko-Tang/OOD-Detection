import torch
# Baseline and ODIN are implemented in detector.py
import argparse
from wideresnet import Wide_ResNet
from detector import Baseline_Detector, ODIN_Detector
import matplotlib.pyplot as plt
from utils import get_cifar10_trainset,get_dataset,test_inference,train_utils


if __name__=='__main__':
    # train WRN28-10
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200,
                        help="number of rounds of training")
    parser.add_argument('--bs', type=int, default=128,
                        help="batch size")
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--reg', default=5e-4, type=float, 
                        help='weight decay for an optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.0)') 
    parser.add_argument('--nest',type=int,default=1,help="Use Nesterov")
    parser.add_argument('--ms',type=int,default=[60,120,160],nargs="*",
                        help="Milestones for learning decay")
    parser.add_argument('--gamma',type = float,default=0.2,
                        help = 'Learning rate decay at milestones')
    
    args = parser.parse_args()
    net = Wide_ResNet(28,10,0.0,10)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    trainloader = get_cifar10_trainset(args.bs,4)
    InDataLoader = get_dataset('cifar10',128,4)

    print("Pretraining...")
    train_utils(args,net,trainloader,InDataLoader,device)

    
    # acc,loss = test_inference(net,InDataLoader,device)
    # print("In-Distribution Accuracy: %f"%acc)

    # dataset = ['SVHN','LSUN','LSUN_resize','MNIST','Uniform','Gaussian']
    # for d in dataset:
    #     OutDataLoader = get_dataset(d,128,4)
    #     BaseDet = Baseline_Detector(net,InDataLoader,OutDataLoader,device)
    #     ODINDet = ODIN_Detector(net,InDataLoader,OutDataLoader,epsilon=0.004,T=1000,IPP=True,device=device)
    #     base_auroc,base_fpr = BaseDet.calculate_metrics()
    #     odin_auroc,odin_fpr = ODINDet.calculate_metrics()
    #     print("Performance on {}:".format(d))
    #     print("\t\tAUROC\tFPR@95TPR")
    #     print("Baseline\t{:.1f}\t{:.1f}".format(base_auroc*100,base_fpr*100))
    #     print("ODIN\t\t{:.1f}\t{:.1f}".format(odin_auroc*100,odin_fpr*100))