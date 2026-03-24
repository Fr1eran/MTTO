import matplotlib.pyplot as plt
import numpy as np


def EnergyRewardFunction(delta_acc):
    # 使用指数衰减式的钟形曲线
    norm_jerk = delta_acc / (0.75)
    return -0.04 * (1 - np.exp(-3.0 * norm_jerk))


delta_acc = np.linspace(0, 2, 200)
reward = EnergyRewardFunction(delta_acc=delta_acc)
plt.plot(delta_acc, reward)
plt.xlabel(r"$\Delta a$")
plt.ylabel(r"$R_{\mathrm{C}}$")
plt.show()
