# How to train

```bash
python hm_main.py --train_rule FedTS --local_bs 64 --dataset cifar --beta 0.5 --teacher_percent 0.4 --backbone resnet18
```

+ `--dataset` can be `cifar`, `cinic10` and `fmnist`. 
+ `--beta` can be `0.5` or `2.0`
+ `--train_rule` can be:
  + `FedTS`
  + `FedAvg`
  + `FedL2Reg`
  + `FedProx`
  + `pFedGraph` 


## Scripts

+ Train on different teacher architectures: `. scripts/train_model_arch.sh`.
+ Train on different teacher percent: `. scripts/train_teacher_percent.sh`.