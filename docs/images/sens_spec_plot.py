import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

plt.style.use("seaborn-v0_8-paper")
plt.style.use("styles/l3_mod.mplstyle")

font = {"fontsize": 16}
alpha = 0.4

fig, ax = plt.subplots(figsize=(8, 5))

x1 = np.linspace(-4, 4, num=200)
p1 = norm.pdf(x1, loc=0, scale=1)

x2 = np.linspace(-1, 11, num=200)
p2 = norm.pdf(x2, loc=5, scale=2)

threshold = 1.2

# Class 0 (Negative)
ax.plot(x1, p1, lw=2)
ax.text(-3.4, 0.35, r"$P(X|Y=0)$", **font)

# Class 1 (Positive)
ax.plot(x2, p2, lw=2)
ax.text(7.0, 0.15, r"$P(X|Y=1)$", **font)

# True Negative - TN
ax.fill_between(x1, p1, where=x1 <= threshold, alpha=alpha, label="TN")
ax.text(-0.3, 0.2, "TN", **font)

# True Positive - TP
ax.fill_between(x2, p2, where=x2 >= threshold, alpha=alpha, label="TP")
ax.text(4.7, 0.1, "TP", **font)

# False Positive - FP
ax.fill_between(x1, p1, where=x1 >= threshold, alpha=alpha, label="FP")
ax.annotate(
    "FP",
    xy=(1.47, 0.07),
    xytext=(2.3, 0.15),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.25"),
    **font,
)

# False Negative - FN
ax.fill_between(x2, p2, where=x2 <= threshold, alpha=alpha, label="FN")
ax.annotate(
    "FN",
    xy=(0.8, 0.01),
    xytext=(-3.0, 0.10),
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.25"),
    **font,
)

# threshold
ax.axvline(threshold, c="red", ls="--", lw=2)
ax.text(threshold + 0.2, 0.39, "Limite", **font)

ax.legend()
ax.set_xlabel(r"Medida $x$")
ax.set_ylabel(r"Probabilidade $P(X|Y)$")
ax.set_xlim((-5.0, 12.0))
ax.set_axis_off()

fig.tight_layout()
fig.savefig("images\\classes_distribution", dpi=600)
plt.show()
