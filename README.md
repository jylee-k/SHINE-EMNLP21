# PQ-GCN

For the pretrained NELL entity embedding and Glove6B word embedding used, you can download them from [here](https://drive.google.com/file/d/1gzIsN6XVqEXPJQR8MXVolbmKqlPgU_YA/view?usp=sharing). 

### Torch Version:
- Python 3.7
- Pytorch 1.2


## Use your own datasets

1. Prepare your data in the form of "{dataset_name}_split.json" in `./data`

2. Update the dataset name at the bottom of `preprocess.py`

3. Run
```
python preprocess.py
```

## Model training
```
cd PQGCN
```


You can choose a specific dataset e.g. ARC by: 
```
python train.py --dataset arc
```
Likewise, you can choose the specific GPU by:
```
python train.py --dataset arc --gpu 2
```








