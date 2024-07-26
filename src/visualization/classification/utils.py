import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import RocCurveDisplay, auc
from src.models.classification.utils import binary_classification
from src.utils.constants import TARGET_NAMES
from typing import Any, Dict, List, Tuple


def roc_analysis(
    results: List[List[Dict[str, Any]]],
    valid_files: List[str],
    model_id: int,
    figsize: Tuple[int] = (15, 5),
    xlim: Tuple[float] = (),
    ylim: Tuple[float] = (),
):
    n_folds = len(results[model_id])

    f, axs = plt.subplots(1, 3, figsize=figsize)

    for label in range(3):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for fold, data in enumerate(results[model_id]):
            model = data["model"]

            data_file = valid_files[fold]
            y, y_hat = binary_classification(model, data_file, label=label)

            viz = RocCurveDisplay.from_predictions(
                y,
                y_hat,
                name=f"ROC fold {fold + 1}",
                alpha=0.3,
                lw=1,
                ax=axs[label],
                plot_chance_level=(fold == n_folds - 1),
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        axs[label].step(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        axs[label].fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm 1 \sigma$",
        )

        axs[label].set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"Mean ROC curve with variability\n(Positive label '{TARGET_NAMES[label]}')",
        )
        axs[label].legend(loc="lower right", prop={"size": 8})
        axs[label].grid()

        if xlim:
            axs[label].set_xlim(xlim)

        if ylim:
            axs[label].set_ylim(ylim)
    f.tight_layout()
    plt.show()
