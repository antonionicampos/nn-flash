import matplotlib.pyplot as plt
import numpy as np


def sp_index(e_0, e_1):
    e_M = (e_0 + e_1) / 2
    e_G = np.sqrt(e_0 * e_1)
    return np.sqrt(e_M * e_G)


e_0 = np.linspace(0, 1, num=1000)
e_1 = np.linspace(0, 1, num=1000)

e_00, e_11 = np.meshgrid(e_0, e_1)

e_sp = sp_index(e_00, e_11)

f, ax = plt.subplots()
h = ax.contourf(e_00, e_11, e_sp, levels=20)
ax.set_xlabel(r"$\hat{e}_0$")
ax.set_ylabel(r"$\hat{e}_1$")

f.colorbar(h, ax=ax)
f.tight_layout()

plt.axis("scaled")
plt.show()
f.savefig("images\\sum-prod.png", dpi=600)

