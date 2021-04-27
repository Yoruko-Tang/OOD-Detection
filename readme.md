# Project 9: Out-of-Distribution Detection

This is an introduction of the codes for reproducing the results in my reports.  I use three pretrained models to evaluate the OOD detectors, including DenseNet-BC, Wide ResNet and ResNet. The pretrained models are stored in the ```./models/``` folder as ```./models/densenet10.pth```, ```./models/wideresnet10.pt``` and ```./models/resnet20.pt``` respectively. The pretrained DenseNet-BC is downloaded  from [here](https://github.com/facebookresearch/odin). The Wide ResNet is trained by myself and the introduction for pretraining a Wide ResNet will be given below. ResNet20 is pretrained in Homework 3 and I use it directly without any pruning or quantization.

### Pretrain a Wide ResNet

The codes for pretraining Wide ResNet are in ```./train_main.py```. To run these codes, use the following command at root.

```shell
CUDA_VISIBLE_DEVICES=0 python3 ./train_main.py
```

You can also change the training hyperparameters by using the options in the command. For more details about the options, see ```./train_main.py```.

### Reproduce the OOD Detection Results

To reproduce the results in the report, refer to ```./main.ipynb``` which is a notebook containing all codes to produce the results of Baseline and ODIN. There are three parts in the notebook:

1. The first part (Block 1 and 2) is used to load the pretrained model. Uncomment the codes that corresponds to the model you want to evaluate with in Part 2 and Part 3.
2. The second part (Block 3 - 14) is used to produce the results of Baseline and ODIN. Each two blocks produce the results on one OOD dataset. See the titles in the notebook for the name of OOD datasets.
3. The third part (Block 15 - 18) evaluates the effects of IPP. Specifically, Block 15 and 16 shows the results on LSUN (crop) with and without IPP. Block 16 and 17 shows the results with different perturbation magnitudes.



