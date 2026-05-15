import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# --- 1. parameter initialization ---
T_p = 430.0  # planned operation time (s)
v_max_kmh = 500.0  # max reference speed (km/h)
v_max_ms = v_max_kmh / 3.6

# initial dynamic parameters
init_K_safe = 1.0
init_K_late = 10.0
init_alpha = 5.0
init_v_act_kmh = 300.0  # actual speed when running late (km/h)
init_delta_s = 100.0  # step distance (m)


# --- 2. core computation ---
def calc_phi_and_diff(t_red_array, K_safe, K_late, alpha, v_act_kmh, delta_s):
    rho = t_red_array / T_p
    v_act_ms = v_act_kmh / 3.6

    # potential function
    phi = np.zeros_like(rho)
    pos_mask = rho >= 0
    neg_mask = rho < 0

    phi[pos_mask] = 1.0 + K_safe * rho[pos_mask]
    phi[neg_mask] = (
        1.0
        + K_safe * rho[neg_mask]
        - (K_late / alpha)
        * (np.exp(-alpha * rho[neg_mask]) + alpha * rho[neg_mask] - 1)
    )

    # differential shaping reward (analytic derivative)
    grad = np.zeros_like(rho)
    grad[pos_mask] = K_safe
    grad[neg_mask] = K_safe + K_late * (np.exp(-alpha * rho[neg_mask]) - 1)

    # physical time calculation
    delta_t_op = delta_s / v_act_ms
    delta_t_min = -(delta_s / v_max_ms)
    delta_t_red = -delta_t_op - delta_t_min
    delta_rho = delta_t_red / T_p

    diff_reward = grad * delta_rho
    return phi, diff_reward


# time redundancy range: 56s to -120s
t_red_arr = np.linspace(56, -120, 500)

# --- 3. UI and plotting ---
fig, (ax_phi, ax_diff) = plt.subplots(2, 1, figsize=(10, 9))
plt.subplots_adjust(left=0.1, bottom=0.35, top=0.95, hspace=0.45)

# initial data
phi_init, diff_init = calc_phi_and_diff(
    t_red_arr, init_K_safe, init_K_late, init_alpha, init_v_act_kmh, init_delta_s
)

# plotting
(line_phi,) = ax_phi.plot(t_red_arr, phi_init, "b-", lw=2)
ax_phi.set_title(
    r"Punctuality potential function ($\Phi_T$ vs time margin)",
)
ax_phi.set_xlabel("Time redundancy (s) [positive = ahead, negative = behind]")
ax_phi.set_ylabel(r"Potential value $\Phi_T$")
ax_phi.invert_xaxis()  # invert x-axis so early (left) vs late (right)
ax_phi.grid(True, linestyle="--", alpha=0.6)

(line_diff,) = ax_diff.plot(t_red_arr, diff_init, "r-", lw=2)
ax_diff.set_title(
    rf"Step-wise dense reward ($R_T$, $\Delta s$={init_delta_s}m)",
)
ax_diff.set_xlabel("Time redundancy (s)")
ax_diff.set_ylabel("Step reward magnitude")
ax_diff.invert_xaxis()
ax_diff.grid(True, linestyle="--", alpha=0.6)

# --- 4. interactive sliders ---
axcolor = "lightgoldenrodyellow"
ax_k_late = plt.axes((0.15, 0.20, 0.65, 0.03), facecolor=axcolor)
ax_alpha = plt.axes((0.15, 0.15, 0.65, 0.03), facecolor=axcolor)
ax_v_act = plt.axes((0.15, 0.10, 0.65, 0.03), facecolor=axcolor)
ax_ds = plt.axes((0.15, 0.05, 0.65, 0.03), facecolor=axcolor)

s_k_late = Slider(ax_k_late, "K_late late penalty", 1.0, 50.0, valinit=init_K_late)
s_alpha = Slider(ax_alpha, "Alpha curvature", 1.0, 10.0, valinit=init_alpha)
s_v_act = Slider(ax_v_act, "Train speed (km/h)", 50.0, 499.0, valinit=init_v_act_kmh)
s_ds = Slider(ax_ds, "Step distance (m)", 10.0, 500.0, valinit=init_delta_s)


def update(val):
    new_phi, new_diff = calc_phi_and_diff(
        t_red_arr, init_K_safe, s_k_late.val, s_alpha.val, s_v_act.val, s_ds.val
    )
    line_phi.set_ydata(new_phi)
    line_diff.set_ydata(new_diff)

    ax_phi.relim()
    ax_phi.autoscale_view()
    ax_diff.relim()
    ax_diff.autoscale_view()
    ax_diff.set_title(rf"Step-wise dense reward ($R_T$, $\Delta s$={s_ds.val:.0f}m)")
    fig.canvas.draw_idle()


s_k_late.on_changed(update)
s_alpha.on_changed(update)
s_v_act.on_changed(update)
s_ds.on_changed(update)

plt.show()
