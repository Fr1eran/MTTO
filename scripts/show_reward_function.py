import matplotlib.pyplot as plt
import numpy as np


def EnergyRewardFunction(delta_acc):
    # 使用指数衰减式的钟形曲线
    norm_jerk = delta_acc / (0.75)
    return -0.04 * (1 - np.exp(-3.0 * norm_jerk))


def DockingRewardFunction(delta_pos):
    # return 2 * np.exp(-2.310490602 * delta_pos) - 1
    # return 4 * np.exp(-0.958940241 * delta_pos) - 3
    return 6 * np.exp(-0.607738523 * delta_pos) - 5


def PunctualityRewardFunction(delta_time):
    return 2 * np.exp(-0.005776227 * delta_time) - 1


# delta_acc = np.linspace(0, 2, 200)
# reward = EnergyRewardFunction(delta_acc=delta_acc)
# plt.plot(delta_acc, reward)
# plt.xlabel(r"$\Delta a$")
# plt.ylabel(r"$R_{\mathrm{C}}$")
# plt.show()

# delta_pos = np.linspace(0, 10, 200)
# reward = DockingRewardFunction(delta_pos=delta_pos)
# plt.plot(delta_pos, reward)
# plt.show()

delta_time = np.linspace(0, 200, 1000)
reward = PunctualityRewardFunction(delta_time=delta_time)
plt.plot(delta_time, reward)
plt.show()
