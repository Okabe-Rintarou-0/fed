import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

fig, ax = plt.subplots()
# Turns off grid on the left Axis.
ax.grid(False)
x_labels = ["ResNet18", "ResNet34", "ResNet50", "ResNet101"]
x_positions = range(len(x_labels))
plt.xticks(x_positions, x_labels)
twin = ax.twinx()
twin.grid(False)

(p1,) = ax.plot(
    x_positions,
    [54.79, 55.41, 56.54, 56.53],
    "C0",
    marker="s",
    # linestyle="--",
    alpha=0.7,
    label="Model Average Accuracy",
)
(p2,) = ax.plot(
    x_positions,
    [42.43, 42.12, 40.90, 41.69],
    "C1",
    marker="D",
    # linestyle="--",
    alpha=0.7,
    label="Student Model Average Accuracy",
)
(p3,) = twin.plot(
    x_positions,
    [11253086, 21368686, 23824767, 42869170, 58558949],
    "C2",
    label="Number of Parameters",
    marker="o",
    # linestyle="--",
    alpha=0.7,
)

ax.set(xlim=(-0.5, len(x_labels) - 0.5), ylabel="Test Accuracy")
twin.set(ylabel="Number of Parameters")

ax.legend(handles=[p1, p2, p3])

plt.savefig("fig2.png")
