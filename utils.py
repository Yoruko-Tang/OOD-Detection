from torchvision import datasets, transforms
from torchvision.utils import save_image,make_grid
import torch.nn.functional as F
import torch
import numpy as np
import os
import glob
import scipy.misc as misc
import matplotlib.pyplot as plt

class Gray2RGB(object):
    def __init__(self):
        pass

    def __call__(self,img):
        return torch.cat([img,img,img],dim = 0)

def Generate_Noise(size = [10000,3,32,32],kind='Gaussian'):
    if kind == 'Gaussian':
        noise = np.clip(np.random.normal(0.5,1.0,size=size),0.0,1.0)
    elif kind == 'Uniform':
        noise = np.random.uniform(0.0,1.0,size=size)
    return noise

def get_cifar10_trainset(batch_size=128,num_workers=1):
    data_dir='./data/cifar10'
    transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
                ])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                      transform=transform)
    DataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=num_workers)
    return DataLoader

def get_dataset(name,batch_size=1,num_workers=1):
    data_dir = './data'
    if name=='cifar10':# 10000x3x32x32
        data_dir+='/cifar10'
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
                ])
        test_dataset=datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=transform)
    elif name=='SVHN':# 26032x3x32x32
        data_dir+='/SVHN'
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
                ])
        test_dataset=datasets.SVHN(data_dir, split='test', download=True,
                                      transform=transform)
    elif name=='LSUN':# 10000x3x36x36
        data_dir+='/LSUN'
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.CenterCrop(32),
                    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
                ])
        test_dataset=datasets.ImageFolder(data_dir,transform=transform)
    elif name=='LSUN_resize':# 10000x3x36x36
        data_dir+='/LSUN_resize'
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    # transforms.CenterCrop(32),
                    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
                ])
        test_dataset=datasets.ImageFolder(data_dir,transform=transform)
    elif name=='MNIST':# 10000x1x28x28
        data_dir+='/MNIST'
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Pad(2),
                    Gray2RGB(),
                    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
                ])
        test_dataset=datasets.MNIST(data_dir, train=False, download=True,
                                      transform=transform)
    elif name=='Uniform' or name == 'Gaussian':
        data_dir=data_dir+'/'+name
        if (not os.path.exists(data_dir+'/Noise')) or len(glob.glob(data_dir+'/Noise/*.jpg'))<10000:
            try:
                os.makedirs(data_dir+'/Noise')
            except OSError:
                pass
            print("Generating Noise Image...")
            noise_img = Generate_Noise(kind=name)
            for i in range(noise_img.shape[0]):
                save_image(torch.tensor(noise_img[i,:,:,:]),data_dir+'/Noise/%d.jpg'%i)
            print("Generating Finished!")
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
                ])
        test_dataset=datasets.ImageFolder(data_dir,transform=transform)
    
    else:
        print("Not supported dataset!")
        raise NotImplementedError
    #print(len(test_dataset))
    DataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)
    return DataLoader
        
def test_inference(model,testloader,device):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = F.cross_entropy

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss/(batch_idx+1)

def train_utils(args,model,trainloader,testloader,device):
    model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,weight_decay=args.reg,
                                momentum=args.momentum,nesterov=args.nest)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,args.ms,args.gamma,verbose=True)
    max_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for (images,labels) in trainloader:
            images,labels = images.to(device),labels.to(device)
            model.zero_grad()
            output = model(images)
            loss = F.cross_entropy(output,labels)
            loss.backward()
            optimizer.step()
        acc,tloss = test_inference(model,testloader,device)
        print("Epoch:{}/{}\tloss:{:.3f}".format(epoch,args.epochs,tloss))
        lr_scheduler.step()
        if acc>max_acc:
            if not os.path.exists('./models'):
                os.makedirs('./models')
            torch.save(model.state_dict(),'./models/wideresnet10.pt')
    
    return model,max_acc



if __name__ == '__main__':
    # dataloader=get_dataset('CIFAR10',8,1)
    # for (img,lab) in dataloader:
    #     print(img.shape)
    #     nping=np.transpose(make_grid(img).numpy(),(1,2,0))
    #     plt.imshow(nping,interpolation='nearest')
    #     plt.show()
    #     break

    indataloader=get_dataset('cifar10',16,4)
    # outdataloader = get_dataset('SVHN',16,1)
    net = torch.load("./models/densenet10.pth")
    a,l = test_inference(net,indataloader,torch.device('cuda'))
    print(a,l)





    