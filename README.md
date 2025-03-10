


## 🚀 Set up 
```
conda create --name torch17 pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
conda activate torch17
pip install -r requirements.txt
```

Register to the competition to access the [**PBVS dataset**](https://codalab.lisn.upsaclay.fr/competitions/17014#participate), and update **dataroot_lr/dataroot_guide/dataroot_gt path** in the json file.

For training on the small network for x16:

    python main_train.py --opt train_baseline.json

For testing:

    python test.py --opt options/test.json

