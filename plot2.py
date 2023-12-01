import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
plt.rcParams.update({"font.size": 20})
fig, ax = plt.subplots()
# Turns off grid on the left Axis.
ax.grid(False)  # 调整X轴标签的字体大小为16
ax.set_ylabel("Y-axis", fontsize=16)

x_labels = ["ResNet18", "ResNet34", "ResNet50", "ResNet101"]
x_positions = range(len(x_labels))
plt.xticks(x_positions, x_labels)
twin = ax.twinx()
twin.grid(False)
twin.set_ylabel("Y-axis", fontsize=16)

(p1,) = ax.plot(
    x_positions,
    [54.79, 55.41, 56.54, 56.53],
    "C0",
    marker="s",
    # linestyle="--",
    linewidth=4,
    markersize=10,
    label=r"Model Average Accuracy",
)
(p2,) = ax.plot(
    x_positions,
    [42.43, 42.12, 40.90, 41.69],
    "C1",
    marker="D",
    # linestyle="--",
    linewidth=4,
    markersize=10,
    label=r"Student Model Average Accuracy",
)
(p3,) = twin.plot(
    x_positions,
    [11253086, 21368686, 23824767, 42869170],
    "C2",
    label="Number of Parameters",
    marker="o",
    # linestyle="--",
    linewidth=4,
    markersize=10,
)

(p4,) = ax.plot(
    x_positions,
    [42.26] * 4,
    marker="p",
    linewidth=4,
    markersize=10,
    color="C3",
    label=r"Baseline(FedAvg)",
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
ax.legend(fontsize=16, handles=[p1, p2, p3, p4], loc="center left", frameon=False)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("fig2.png", dpi=600, bbox_inches="tight")
