betas=("0.5" "2.0")

for beta in "${betas[@]}"; do
    for ((num=19; num>=0; num--)); do
        python hm_main.py --train_rule FedTS --local_bs 64 \
        --teacher_percent 0 \
        --num_clients 20 \
        --stu_idx "$num" \
        --dataset cifar \
        --beta "$beta"
    done
done

