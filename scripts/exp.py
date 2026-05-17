import matplotlib.pyplot as plt
import numpy as np

from utils.plot_utils import set_global_plot_style

set_global_plot_style(
    font_preset="sci",
    preferred_font="Calibri",
    title_font_size=12.0,
    axis_label_font_size=12.0,
    tick_font_size=12.0,
    legend_font_size=12.0,
    figure_dpi=150.0,
    savefig_dpi=300.0,
)

x = np.linspace(-2.0, 0.0, 100)
y = np.exp(x)

fig = plt.figure()
ax = fig.add_subplot()

ax.plot(x, y)
ax.set_xlim(-2.0, 0.0)

plt.show()
