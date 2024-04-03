python hm_main.py --local_bs 64 --train_rule FedTS --backbone resnet34 --dataset cifar --device cuda --teacher_percent 0.4
python hm_main.py --local_bs 64 --train_rule FedTS --backbone resnet50 --dataset cifar --device cuda --teacher_percent 0.4
python hm_main.py --local_bs 64 --train_rule FedTS --backbone resnet101 --dataset cifar --device cuda --teacher_percent 0.4
python hm_main.py --local_bs 64 --train_rule FedTS --backbone resnet152 --dataset cifar --device cuda --teacher_percent 0.4
python hm_main.py --local_bs 64 --train_rule FedTS --backbone resnet34 --dataset cifar --device cuda --teacher_percent 0.4 --beta 2
python hm_main.py --local_bs 64 --train_rule FedTS --backbone resnet50 --dataset cifar --device cuda --teacher_percent 0.4 --beta 2
python hm_main.py --local_bs 64 --train_rule FedTS --backbone resnet101 --dataset cifar --device cuda --teacher_percent 0.4 --beta 2
python hm_main.py --local_bs 64 --train_rule FedTS --backbone resnet152 --dataset cifar --device cuda --teacher_percent 0.4 --beta 2