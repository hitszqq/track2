


## 🚀 Set up 
```
conda create --name torch17 pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
conda activate torch17
pip install -r requirements.txt
```

Register to the competition to access the [**PBVS dataset**](https://codalab.lisn.upsaclay.fr/competitions/17014#participate), and update **dataroot_lr/dataroot_guide/dataroot_gt path** in the json file.

For training on the small network for x8 guided super resolution(944,491 parameters, trainable with one RTX 3080 12GB):

    python main_train_SwinFuSR.py --opt options/train_baseline.json

For training on the large network for x8 guided super resolution(3.3 M parameters, trainable with two V100 32GB):

    python main_train_SwinFuSR.py --opt options/train_final.json

