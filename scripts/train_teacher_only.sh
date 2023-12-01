num_clients_list=("4" "8")
betas=("0.5" "2.0")

for beta in "${betas[@]}"; do
    for num_clients in "${num_clients_list[@]}"; do
        python hm_main.py --train_rule FedTS --local_bs 64 \
        --teacher_percent 1.0 \
        --device cuda \
        --num_clients "$num_clients" \
        --dataset cifar \
        --beta "$beta"
    done
done