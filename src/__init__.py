import matplotlib
matplotlib.use("agg")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
plt.style.use(os.path.join("src", "visualization", "styles", "l3_mod.mplstyle"))
