import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

x = list(range(0, 21))

a_stu = list(range(0, 21))
a = list(range(0, 21))

a_stu_beta_2 = list(range(0, 21))
a_beta_2 = list(range(0, 21))

plt.plot(x, a_stu, label=r'Student Model Average Accuracy($\beta=0.5$)')
plt.plot(x, a_stu_beta_2, label=r'Student Model Average Accuracy($\beta=2.0$)')
plt.plot(x, a, label=r'Model Average Accuracy($\beta=0.5$)')
plt.plot(x, a_beta_2, label=r'Model Average Accuracy($\beta=2.0$)')

plt.xticks(x)
plt.legend()
plt.savefig("plot3.png")
