teacher_percents=("0.2" "0.4")
rules=("FedTS" "FedL2Reg" "FedProx" "pFedGraph")

for rule in "${rules[@]}"; do
    for teacher_percent in "${teacher_percents[@]}"; do
        python hm_main.py --train_rule "$rule" --local_bs 64 \
        --teacher_percent "$teacher_percent" \
        --dataset fmnist \
        --device cpu \
        --beta 2.0
    done
done