betas=("0.5" "2.0")

for beta in "${betas[@]}"; do
    for ((num=0; num<=20; num++)); do
        teacher_percent=$(printf "%.2f" $(echo "scale=2; $num/20" | bc))
        python hm_main.py --train_rule FedTS --local_bs 64 \
        --device cuda \
        --teacher_percent "$teacher_percent" \
        --num_clients "$num" \
        --dataset cifar \
        --beta "$beta"
    done
done

