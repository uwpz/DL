
# ######################################################################################################################
# Libraries
# ######################################################################################################################

# Data
import numpy as np
import pandas as pd

# Plot
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# ETL
from scipy.stats import chi2_contingency
from scipy.sparse import hstack
from scipy.cluster.hierarchy import ward, dendrogram, fcluster

# ML
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.calibration import calibration_curve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


# Util
from collections import defaultdict
import os
from os import getcwd
import pdb  # pdb.set_trace()  #quit with "q", next line with "n", continue with "c"
from joblib import Parallel, delayed
from dill import (load_session, dump_session)
import pickle


# ######################################################################################################################
# Parameters
# ######################################################################################################################

# Locations
plotloc = "./output/"

# Util
sns.set(style="whitegrid")
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)
# plt.ioff(); plt.ion()  # Interactive plotting? ion is default

# Other
twocol = ["red", "green"]


# ######################################################################################################################
# My Functions and Classes
# ######################################################################################################################

# Plot fit history
def plot_fit(fit):
    epochs = 1 + np.arange(len(fit.history["acc"]))
    fig, ax = plt.subplots(1,2)
    ax[0].plot(epochs, fit.history["val_acc"], "-o", label="Validation")
    ax[0].plot(epochs, fit.history["acc"], "-o", label="Training")
    ax[0].legend()
    ax[0].set_title("Acc")
    ax[1].plot(epochs, fit.history["val_loss"], "-o", label="Validation")
    ax[1].plot(epochs, fit.history["loss"], "-o", label="Training")
    ax[1].legend()
    ax[1].set_title("Loss")
    fig.tight_layout()
    plt.show()


def plot_cam(model, img_path, img_idx, yhat, y,
           nrow=2, ncol=2, w=18, h=12, pdf=None):
#model=model1; img_path=dataloc + "test/"; img_idx=i_img; ncol=4; nrow=3; w=12; h=8; pdf=None

    # Get files
    files = []
    for r, d, f in os.walk(img_path):
        for file in f:
            if ".jpg" in file:
                files.append(os.path.join(r, file))

    # Open pdf
    if pdf is not None:
        pdf_pages = PdfPages(pdf)

    # Plot
    n_ppp = ncol * nrow
    for i in range(len(img_idx)):
        if (i*2) % n_ppp == 0:
            fig, ax = plt.subplots(nrow, ncol)
            fig.set_size_inches(w=w, h=h)
            fig.tight_layout()
            i_ax = 0

        # Image
        img_idx_act = img_idx[i]
        img = image.img_to_array(image.load_img(files[img_idx_act], target_size=targetsize)) / 255
        ax_act = ax.flat[i_ax]
        ax_act.imshow(img)
        ax_act.set_title("Class = " + str(y[img_idx_act]))
        ax_act.set_axis_off()
        i_ax += 1

        # Heatmap
        ax_act = ax.flat[i_ax]
        grads = visualize_cam(model, layer_idx=len(model.layers) - 1, filter_indices=0, seed_input=img) / 255
        ax_act.imshow(0.5 * img + 0.5 * grads)
        ax_act.set_title("yhat = " + str(yhat[img_idx_act, 1].round(3)))
        ax_act.set_axis_off()
        i_ax += 1

        if i_ax == n_ppp or i == len(img_idx)-1:
            pdf_pages.savefig(fig)

    if pdf is not None:
        pdf_pages.close()


# Plot ML-algorithm performance
def plot_all_performances(y, yhat, target_type="CLASS", ylim=None, w=18, h=12, pdf=None):
    # y=df_test["target"]; yhat=yhat_test; ylim = None; w=12; h=8
    fig, ax = plt.subplots(2, 3)

    if target_type == "CLASS":
        # Roc curve
        ax_act = ax[0, 0]
        fpr, tpr, cutoff = roc_curve(y, yhat[:, 1])
        roc_auc = roc_auc_score(y, yhat[:, 1])
        sns.lineplot(fpr, tpr, ax=ax_act, palette=sns.xkcd_palette(["red"]))
        props = {'xlabel': r"fpr: P($\^y$=1|$y$=0)",
                 'ylabel': r"tpr: P($\^y$=1|$y$=1)",
                 'title': "ROC (AUC = {0:.2f})".format(roc_auc)}
        ax_act.set(**props)

        # Confusion matrix
        ax_act = ax[0, 1]
        df_conf = pd.DataFrame(confusion_matrix(y, np.where(yhat[:, 1] > 0.5, 1, 0)))
        acc = accuracy_score(y, np.where(yhat[:, 1] > 0.5, 1, 0))
        sns.heatmap(df_conf, annot=True, fmt=".5g", cmap="Greys", ax=ax_act)
        props = {'xlabel': "Predicted label",
                 'ylabel': "True label",
                 'title': "Confusion Matrix (Acc ={0: .2f})".format(acc)}
        ax_act.set(**props)

        # Distribution plot
        ax_act = ax[0, 2]
        sns.distplot(yhat[:, 1][y == 1], color="red", label="1", bins=20, ax=ax_act)
        sns.distplot(yhat[:, 1][y == 0], color="blue", label="0", bins=20, ax=ax_act)
        props = {'xlabel': r"Predictions ($\^y$)",
                 'ylabel': "Density",
                 'title': "Distribution of Predictions",
                 'xlim': (0, 1)}
        ax_act.set(**props)
        ax_act.legend(title="Target", loc="best")

        # Calibration
        ax_act = ax[1, 0]
        true, predicted = calibration_curve(y, yhat[:, 1], n_bins=10)
        sns.lineplot(predicted, true, ax=ax_act, marker="o")
        props = {'xlabel': r"$\bar{\^y}$ in $\^y$-bin",
                 'ylabel': r"$\bar{y}$ in $\^y$-bin",
                 'title': "Calibration"}
        ax_act.set(**props)

        # Precision Recall
        ax_act = ax[1, 1]
        prec, rec, cutoff = precision_recall_curve(y, yhat[:, 1])
        prec_rec_auc = average_precision_score(y, yhat[:, 1])
        sns.lineplot(rec, prec, ax=ax_act, palette=sns.xkcd_palette(["red"]))
        props = {'xlabel': r"recall=tpr: P($\^y$=1|$y$=1)",
                 'ylabel': r"precision: P($y$=1|$\^y$=1)",
                 'title': "Precision Recall Curve (AUC = {0:.2f})".format(prec_rec_auc)}
        ax_act.set(**props)
        for thres in np.arange(0.1, 1, 0.1):
            i_thres = np.argmax(cutoff > thres)
            ax_act.annotate("{0: .1f}".format(thres), (rec[i_thres], prec[i_thres]), fontsize=10)

        # Precision
        ax_act = ax[1, 2]
        pct_tested = np.array([])
        for thres in cutoff:
            pct_tested = np.append(pct_tested, [np.sum(yhat[:, 1] >= thres)/len(yhat)])
        sns.lineplot(pct_tested, prec[:-1], ax=ax_act, palette=sns.xkcd_palette(["red"]))
        props = {'xlabel': "% Samples Tested",
                 'ylabel': r"precision: P($y$=1|$\^y$=1)",
                 'title': "Precision Curve"}
        ax_act.set(**props)
        for thres in np.arange(0.1, 1, 0.1):
            i_thres = np.argmax(cutoff > thres)
            ax_act.annotate("{0: .1f}".format(thres), (pct_tested[i_thres], prec[i_thres]), fontsize=10)

    if target_type == "REGR":
        def plot_scatter(x, y, xlabel="x", ylabel="y", title=None, ylim=None, ax_act=None):
            if ylim is not None:
                ax_act.set_ylim(ylim)
                tmp_scale = (ylim[1] - ylim[0]) / (np.max(y) - np.min(y))
            else:
                tmp_scale = 1
            tmp_cmap = colors.LinearSegmentedColormap.from_list("wh_bl_yl_rd",
                                                                [(1, 1, 1, 0), "blue", "yellow", "red"])
            p = ax_act.hexbin(x, y,
                              gridsize=(int(50 * tmp_scale), 50),
                              cmap=tmp_cmap)
            plt.colorbar(p, ax=ax_act)
            sns.regplot(x, y, lowess=True, scatter=False, color="black", ax=ax_act)
            ax_act.set_title(title)
            ax_act.set_ylabel(ylabel)
            ax_act.set_xlabel(xlabel)

            ax_act.set_facecolor('white')
            # ax_act.grid(False)

            ylim = ax_act.get_ylim()
            xlim = ax_act.get_xlim()

            # Inner Histogram on y
            ax_act.set_xlim(xlim[0] - 0.3 * (xlim[1] - xlim[0]))
            inset_ax = ax_act.inset_axes([0, 0, 0.2, 1])
            inset_ax.set_axis_off()
            ax_act.get_shared_y_axes().join(ax_act, inset_ax)
            sns.distplot(y, color="grey", vertical=True, ax=inset_ax)

            # Inner-inner Boxplot on y
            xlim_inner = inset_ax.get_xlim()
            inset_ax.set_xlim(xlim_inner[0] - 0.3 * (xlim_inner[1] - xlim_inner[0]))
            inset_inset_ax = inset_ax.inset_axes([0, 0, 0.2, 1])
            inset_inset_ax.set_axis_off()
            inset_ax.get_shared_y_axes().join(inset_ax, inset_inset_ax)
            sns.boxplot(y, palette=["grey"], orient="v", ax=inset_inset_ax)

            # Inner Histogram on x
            ax_act.set_ylim(ylim[0] - 0.3 * (ylim[1] - ylim[0]))
            inset_ax = ax_act.inset_axes([0, 0, 1, 0.2])
            inset_ax.set_axis_off()
            ax_act.get_shared_x_axes().join(ax_act, inset_ax)
            sns.distplot(x, color="grey", ax=inset_ax)

            # Inner-inner Boxplot on x
            ylim_inner = inset_ax.get_ylim()
            inset_ax.set_ylim(ylim_inner[0] - 0.3 * (ylim_inner[1] - ylim_inner[0]))
            inset_inset_ax = inset_ax.inset_axes([0, 0, 1, 0.2])
            inset_inset_ax.set_axis_off()
            inset_ax.get_shared_x_axes().join(inset_ax, inset_inset_ax)
            sns.boxplot(x, palette=["grey"], ax=inset_inset_ax)

            ax_act.set_xlim(xlim[0] - 0.3 * (xlim[1] - xlim[0]))  # need to set again

        # Scatter plots
        plot_scatter(yhat, y,
                     xlabel=r"$\^y$", ylabel="y",
                     title=r"Observed vs. Fitted ($\rho_{Spearman}$ = " +
                           str(spearman_loss_func(y, yhat).round(3)) + ")",
                     ylim=ylim, ax_act=ax[0, 0])
        plot_scatter(yhat, y - yhat,
                     xlabel=r"$\^y$", ylabel=r"y-$\^y$", title="Residuals vs. Fitted",
                     ylim=ylim, ax_act=ax[1, 0])
        plot_scatter(yhat, abs(y - yhat),
                     xlabel=r"$\^y$", ylabel=r"|y-$\^y$|", title="Absolute Residuals vs. Fitted",
                     ylim=ylim, ax_act=ax[1, 1])
        plot_scatter(yhat, abs(y - yhat) / abs(y),
                     xlabel=r"$\^y$", ylabel=r"|y-$\^y$|/|y|", title="Relative Residuals vs. Fitted",
                     ylim=ylim, ax_act=ax[1, 2])

        # Calibration
        ax_act = ax[0, 1]
        df_calib = pd.DataFrame({"y": y, "yhat": yhat})\
            .assign(bin=lambda x: pd.qcut(x["yhat"], 10, duplicates="drop").astype("str"))\
            .groupby(["bin"], as_index=False).agg("mean")\
            .sort_values("yhat")
        sns.lineplot("yhat", "y", data=df_calib, ax=ax_act, marker="o")
        props = {'xlabel': r"$\bar{\^y}$ in $\^y$-bin",
                 'ylabel': r"$\bar{y}$ in $\^y$-bin",
                 'title': "Calibration"}
        ax_act.set(**props)

        # Distribution
        ax_act = ax[0, 2]
        sns.distplot(y, color="blue", label="y", ax=ax_act)
        sns.distplot(yhat, color="red", label=r"$\^y$", ax=ax_act)
        ax_act.set_ylabel("density")
        ax_act.set_xlabel("")
        ax_act.set_title("Distribution")

        ylim = ax_act.get_ylim()
        ax_act.set_ylim(ylim[0] - 0.3 * (ylim[1] - ylim[0]))
        inset_ax = ax_act.inset_axes([0, 0, 1, 0.2])
        inset_ax.set_axis_off()
        ax_act.get_shared_x_axes().join(ax_act, inset_ax)
        df_distr = pd.concat([pd.DataFrame({"type": "y", "values": y}),
                              pd.DataFrame({"type": "yhat", "values": yhat})])
        sns.boxplot(x=df_distr["values"],
                    y=df_distr["type"].astype("category"),
                    # order=df[feature_act].value_counts().index.values[::-1],
                    palette=["blue", "red"],
                    ax=inset_ax)
        ax_act.legend(title="", loc="best")

    # Adapt figure
    fig.set_size_inches(w=w, h=h)
    fig.tight_layout()
    if pdf is not None:
        fig.savefig(pdf)
        # plt.close(fig)
    plt.show()
