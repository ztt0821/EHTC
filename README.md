# EHTC
This is the code of Fish Image Instance Segmentation: An Enhanced Hybrid Task Cascade Approach which is accepted by IEEE ICSC 2021.
We follow the code of https://github.com/open-mmlab/mmdetection and https://github.com/DmitryUlyanov/deep-image-prior.

## `Step 1`: 
install mmdetection. The current version of mmdetection in this link is little different from ours code, because we didn't update it. So if you want to run our code, please install follow the previous version. Here is the link https://github.com/ztt0821/mmdetection/blob/master/docs/install.md. Also here is the link about how to start:https://github.com/ztt0821/mmdetection/blob/master/docs/getting_started.md. Our running environment can be seen in requirements_ehtc.txt

## `Step 2`: 
prepare the dataset. We use two datasets. one is original data, another is super_resolution dataset(after data pre-processing). We use deep image prior to do the data pre-processing. The original fish data is from [fish4knowledge](https://groups.inf.ed.ac.uk/f4k/GROUNDTRUTH/RECOG/). We follow these steps to convert the mask into coco style https://github.com/waspinator/pycococreator. All dataset can be downloaded from here. [original_data](https://drive.google.com/drive/folders/18fNF2JOZMP7hThYBFpvHwdDs33BhK9Y6?usp=sharing), [super_resolution](https://drive.google.com/drive/folders/1o7-kT-VmzrweSjIZZhYf-nNbZA_M78Ph?usp=sharing)

## `Step 3`:
train and test the network like the example in https://github.com/ztt0821/mmdetection/blob/master/docs/getting_started.md. We use only one 2080Ti card to train and test, so if you want to train on multi-gpu, please change the learning rate. Also please change the dataset root in configs/htc/*** files. Because we only enhance the HTC model, so we didn't create ehtc name in configs, just use htc. Therefore, only following files are about EHTC.

go to configs/htc/
 
htc_without_rx32_newdata_semantic_r50_1x.py         ehtc  

htc_without_rx64_newdata_semantic_r50_1x.py         ehtc   [model](https://drive.google.com/file/d/18vPhHvfcZMECnYoaLhKtY8BHu3KzaBJ1/view?usp=sharing)

htc_without_rx64_origdata_semantic_fpn_1x.py        ehtc   [model](https://drive.google.com/file/d/1FUo9Kg540DbuvOkmMCDlm-ulD5_ZCE2g/view?usp=sharing)

other files are about HTC.

Tips:
We draw the PR curve in the paper, if you want to draw it, you only need to modify the mmdet/datasets/coco.py file(you can refer to the mmdet/datasets/coco_example.py)
