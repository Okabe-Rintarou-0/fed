betas=("0.5" "2.0")

for beta in "${betas[@]}"; do
    for ((num=1; num<20; num++)); do
        python hm_main.py --train_rule FedTS --local_bs 64 \
        --device cuda \
        --teacher_percent 1 \
        --num_clients "$num" \
        --dataset cifar \
        --beta "$beta"
    done
done

