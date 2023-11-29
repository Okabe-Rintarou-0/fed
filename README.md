# How to train

```bash
python hm_main.py --train_rule FedTS --local_bs 64 --dataset cifar --beta 0.5
```

+ `--dataset` can be `cifar`, `cinic10` and `fmnist`. 
+ `--beta` can be `0.5` or `2.0`
+ `--train_rule` can be:
  + `FedTS`
  + `FedAvg`
  + `FedL2Reg`
  + `FedProx`
  + `pFedGraph` 