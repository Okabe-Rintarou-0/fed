import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

x = list(range(0, 20))

# a_stu = [
#     44.55,
#     44.86,
#     44.52,
#     43.54,
#     41.53,
#     39.83,
#     40.02,
#     40.96,
#     42.43,
#     42.44,
#     41.18,
#     39.18,
#     36.65,
#     32.62,
#     32.83,
#     35.87,
#     36.75,
#     32.40,
#     22.59,
#     12.95,
# ]
# a = [
#     44.55,
#     46.21,
#     46.63,
#     48.03,
#     47.94,
#     48.29,
#     49.93,
#     51.41,
#     54.79,
#     55.75,
#     57.96,
#     58.56,
#     59.99,
#     61.33,
#     63.86,
#     67.43,
#     69.12,
#     71.00,
#     71.42,
#     73.89,
# ]

a_stu_beta_2 = [
    46.96,
    47.16,
    46.79,
    46.7,
    47.41,
    46.27,
    46.72,
    47.46,
    46.69,
    46.47,
    45.25,
    47.30,
    45.82,
    45.29,
    45.01,
    44.86,
    43.72,
    43.42,
    38.09,
    35.28,
]
a_beta_2 = [
    46.96,
    48.09,
    49.21,
    50.21,
    52.6,
    53.74,
    55.61,
    57.98,
    58.79,
    60.88,
    61.72,
    64.08,
    65.73,
    67.20,
    69.13,
    70.90,
    72.18,
    74.11,
    75.31,
    76.99,
]
baseline = [46.39] * 20
# plt.plot(x, a, marker="o", label="Model Average Accuracy")
plt.plot(x, a_beta_2, marker="o", label="Model Average Accuracy")
# plt.plot(x, a_stu, marker="D", label="Student Model Average Accuracy")

plt.plot(x, a_stu_beta_2, marker="D", label="Student Model Average Accuracy")


plt.plot(x, baseline, label="Baseline(FedAvg)", linestyle="--")
plt.ylabel("Test Accuracy", fontsize=16)
plt.ylabel("Number of Teachers", fontsize=16)
plt.xticks(x)
plt.legend(fontsize=16)
plt.tick_params(axis="x", labelsize=16)
plt.tick_params(axis="y", labelsize=16)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("plot3.png", dpi=600, bbox_inches="tight")
