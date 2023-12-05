import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser

sns.set_theme()
plt.rcParams.update({"font.size": 20})
fig, ax = plt.subplots()
# Turns off grid on the left Axis.
ax.grid(False)  # 调整X轴标签的字体大小为16
ax.set_ylabel("Y-axis", fontsize=16)

x_labels = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
x_positions = range(len(x_labels))
plt.xticks(x_positions, x_labels)
twin = ax.twinx()
twin.grid(False)
twin.set_ylabel("Y-axis", fontsize=16)

parser = ArgumentParser()
parser.add_argument("--beta", default=0.5)
args = parser.parse_args()

beta = args.beta

if beta == 0.5:
    A = [54.79, 55.48, 56.66, 56.47, 56.76]
    A_stu = [42.43, 42.28, 41.32, 41.36, 41.89]
else:
    A = [58.79, 59.45, 60.99, 60.3, 60.51]
    A_stu = [46.69, 45.92, 46.05, 45.69, 45.38]

(p1,) = ax.plot(
    x_positions,
    A,
    "C0",
    marker="s",
    # linestyle="--",
    linewidth=4,
    markersize=10,
    label=r"Model Average Accuracy",
)
(p2,) = ax.plot(
    x_positions,
    A_stu,
    "C1",
    marker="D",
    # linestyle="--",
    linewidth=4,
    markersize=10,
    label=r"Student Model Average Accuracy",
)
(p3,) = twin.plot(
    x_positions,
    [11253086, 21368686, 23824767, 42869170, 58558949],
    "C2",
    label="Number of Parameters",
    marker="o",
    # linestyle="--",
    linewidth=4,
    markersize=10,
)


# ax.plot(
#     x_positions,
#     [48.64] * 4,
#     linestyle="--",
#     color="gray",
#     label=r"Baseline(FedAvg, $\beta=2.0$)",
# )

ax.set(xlim=(-0.5, len(x_labels) - 0.5), ylabel="Test Accuracy")
twin.set(ylabel="Number of Parameters")
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
twin.tick_params(axis="y", labelsize=16)
ax.legend(fontsize=16, handles=[p1, p2, p3], loc="center left", frameon=False)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("plot2.png", dpi=600, bbox_inches="tight")
