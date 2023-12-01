backbones=("resnet18" "resnet34" "resnet50" "resnet101" "resnet152")
betas=("0.5" "2.0")

for beta in "${betas[@]}"; do
    for backbone in "${backbones[@]}"; do
        python hm_main.py --train_rule FedTS --local_bs 64 \
        --backbone "$backbone" \
        --teacher_percent 0.4 \
        --dataset cifar \
        --beta "$beta"
    done
done