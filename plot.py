import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
path = "./data/csv"

labels_map = {
    "FedTS(Ours)": "data/csv/FedTSGen_non_iid_model_het_entropy_agg_fmnist_ta_0.0_te_0.4_new_server_acc_avg_student_acc1.csv",
    "FedProx": "data/csv/FedProx_non_iid_model_het_fmnist_ta_0.0_te_0.4_server_acc_avg_student_acc1.csv",
    "FedL2Reg": "data/csv/FedL2Reg_non_iid_model_het_fmnist_ta_0.0_te_0.4_server_acc_avg_student_acc1.csv",
    "FedAvg": "data/csv/FedAvg_non_iid_fmnist_ta_0.0_te_0.0_server_acc_avg_student_acc1.csv",
    "FedAvg-TS": "data/csv/FedAvg_non_iid_model_het_fmnist_ta_0.0_te_0.4_3_server_acc_avg_student_acc1.csv",
}

for label in labels_map:
    path = labels_map[label]

    df = pd.read_csv(path)

    index = list(df["Value"].index)
    values = df["Value"].values
    values /= 100
    
    plt.plot(index, values, label=label, alpha=0.7)

plt.ylim((0.65, 0.85))
plt.yticks(np.arange(0.65, 0.85, 0.05))
plt.xlabel("Number of local epochs")
plt.ylabel("Test accuracy")
plt.legend()
plt.savefig("./plot1.png")