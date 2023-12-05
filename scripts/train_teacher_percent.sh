betas=("0.5" "2.0")

for beta in "${betas[@]}"; do
    for ((num=0; num<=20; num++)); do
        if [ "$num" -eq 4 ] || [ "$num" -eq 8 ]; then
            continue  # Skip the rest of the loop for num=4 or num=8
        fi
        teacher_percent=$(printf "%.2f" $(echo "scale=2; $num/20" | bc))
        python hm_main.py --train_rule FedTS --local_bs 64 \
        --device cuda \
        --teacher_percent "$teacher_percent" \
        --dataset cifar \
        --beta "$beta"
    done
done

