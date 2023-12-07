import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

x = list(range(1, 20))

a_teacher = [
    71.86,
    65.62,
    73.47,
    73.58,
    73.67,
    73.05,
    70.82,
    73.33,
    72.02,
    74.74,
    74.42,
    75.55,
    76.79,
    77.16,
    77.95,
    77.21,
    77.81,
    76.85,
    77.10,
]
a_teacher_only = [
    71.18,
    67.54,
    74.57,
    74.32,
    73.83,
    74.08,
    72.76,
    73.82,
    72.94,
    75.03,
    75.00,
    76.09,
    77.46,
    77.51,
    78.02,
    77.45,
    78.44,
    76.85,
    77.10,
]

# a_teacher_beta_2 = [
#     46.96,
#     47.16,
#     46.79,
#     46.7,
#     47.41,
#     46.27,
#     46.72,
#     47.46,
#     46.69,
#     46.47,
#     45.25,
#     47.30,
#     45.82,
#     45.29,
#     45.01,
#     44.86,
#     43.72,
#     43.42,
#     38.09,
#     35.28,
# ]
# a_teacher_only_beta_2 = [
#     46.96,
#     48.09,
#     49.21,
#     50.21,
#     52.6,
#     53.74,
#     55.61,
#     57.98,
#     58.79,
#     60.88,
#     61.72,
#     64.08,
#     65.73,
#     67.20,
#     69.13,
#     70.90,
#     72.18,
#     74.11,
#     75.31,
#     76.99,
# ]

# plt.plot(x, a, marker="o", label="Model Average Accuracy")
plt.plot(x, a_teacher, marker="o", label="Teacher Model Average Accuracy")
# plt.plot(x, a_stu, marker="D", label="Student Model Average Accuracy")

plt.plot(
    x, a_teacher_only, marker="D", label="Teacher Model Average Accuracy(w/o students)"
)

plt.ylabel("Test Accuracy", fontsize=16)
plt.xlabel("Number of Teachers", fontsize=16)
plt.xticks(x)
plt.legend(fontsize=16)
plt.tick_params(axis="x", labelsize=16)
plt.tick_params(axis="y", labelsize=16)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("plot4.png", dpi=600, bbox_inches="tight")