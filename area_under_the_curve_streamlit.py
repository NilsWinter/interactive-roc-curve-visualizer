import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st
import io
import pandas as pd

st.set_page_config(layout="wide")
st.title("Interactive ROC Curve Visualizer")

# --- Core Calculation Function ---
def compute_distributions_and_metrics(auc, prevalence, n, threshold, sigma_pos=1, sigma_neg=1):
    z_auc = norm.ppf(auc)
    mu_neg = 0
    mu_pos = z_auc * np.sqrt(sigma_pos ** 2 + sigma_neg ** 2)

    TPR = 1 - norm.cdf((threshold - mu_pos) / sigma_pos)
    TNR = norm.cdf((threshold - mu_neg) / sigma_neg)
    FPR = 1 - TNR
    FNR = 1 - TPR

    n_pos = int(n * prevalence)
    n_neg = n - n_pos

    TP = TPR * n_pos
    FN = FNR * n_pos
    TN = TNR * n_neg
    FP = FPR * n_neg

    lower = min(mu_neg - 4 * sigma_neg, mu_pos - 4 * sigma_pos)
    upper = max(mu_neg + 4 * sigma_neg, mu_pos + 4 * sigma_pos)
    x_vals = np.linspace(lower, upper, 1000)
    y_pos_pdf = norm.pdf(x_vals, loc=mu_pos, scale=sigma_pos) * prevalence
    y_neg_pdf = norm.pdf(x_vals, loc=mu_neg, scale=sigma_neg) * (1 - prevalence)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    balanced_accuracy = (TPR + TNR) / 2
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1_score = (2 * precision * TPR) / (precision + TPR) if (precision + TPR) > 0 else 0

    return {
        "TPR": TPR, "TNR": TNR, "FPR": FPR, "FNR": FNR,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "x_vals": x_vals,
        "y_pos_pdf": y_pos_pdf,
        "y_neg_pdf": y_neg_pdf,
        "mu_pos": mu_pos,
        "mu_neg": mu_neg,
        "sigma_pos": sigma_pos,
        "sigma_neg": sigma_neg,
        "auc_score": auc,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "f1_score": f1_score
    }

# --- Plotting Function ---
def generate_combined_plot(metrics, threshold):
    fig, (ax_kde, ax_roc) = plt.subplots(1, 2, figsize=(12, 6), dpi=200)

    # KDE plot
    ax_kde.plot(metrics["x_vals"], metrics["y_pos_pdf"], label='Positive', color='#c0392b')
    ax_kde.plot(metrics["x_vals"], metrics["y_neg_pdf"], label='Negative', color='#2980b9')
    ax_kde.fill_between(metrics["x_vals"], 0, metrics["y_pos_pdf"],
                        where=(metrics["x_vals"] >= threshold),
                        color='#c0392b', alpha=0.3, label='True Positives')
    ax_kde.fill_between(metrics["x_vals"], 0, metrics["y_neg_pdf"],
                        where=(metrics["x_vals"] >= threshold),
                        color='#2980b9', alpha=0.3, label='False Positives')
    ax_kde.axvline(threshold, color='black', linestyle='--', label='Threshold')
    ax_kde.set_title("Score Distributions")
    ax_kde.set_xlabel("Score")
    ax_kde.set_ylabel("Density")
    ax_kde.legend()

    # ROC curve
    thresholds = np.linspace(metrics["x_vals"].min(), metrics["x_vals"].max(), 500)
    tpr_vals = 1 - norm.cdf((thresholds - metrics["mu_pos"]) / metrics["sigma_pos"])
    fpr_vals = 1 - norm.cdf((thresholds - metrics["mu_neg"]) / metrics["sigma_neg"])

    ax_roc.fill_between(fpr_vals, tpr_vals, step='mid', alpha=0.2, color='gray', label='AUC Area')
    ax_roc.plot(fpr_vals, tpr_vals, label='ROC Curve', color='black')
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance (AUC=0.5)')
    ax_roc.scatter(metrics["FPR"], metrics["TPR"], color='red', s=80, label='Current Threshold')
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf

# --- Sidebar inputs ---
auc = st.sidebar.slider("Desired AUC", min_value=0.5, max_value=0.99, value=0.85, step=0.01)
prevalence = st.sidebar.slider("Prevalence (positive class)", min_value=0.01, max_value=0.99, value=0.5, step=0.01)
n = st.sidebar.number_input("Number of Samples", min_value=100, value=5000, step=100)
sigma_pos = st.sidebar.number_input("Sigma (positive class)", min_value=0.1, value=1.0, step=0.1)
sigma_neg = st.sidebar.number_input("Sigma (negative class)", min_value=0.1, value=1.0, step=0.1)

# Author info
st.sidebar.markdown("---")
st.sidebar.markdown("""
# About the Author  
**Nils R. Winter, PhD**  
[Website](https://nilsrwinter.com)  
[GitHub](https://github.com/NilsWinter)
""")

# --- Threshold slider based on range ---
z_auc = norm.ppf(auc)
mu_neg = 0
mu_pos = z_auc * np.sqrt(sigma_pos**2 + sigma_neg**2)
lower = min(mu_neg - 4 * sigma_neg, mu_pos - 4 * sigma_pos)
upper = max(mu_neg + 4 * sigma_neg, mu_pos + 4 * sigma_pos)

threshold = st.slider(
    "Decision Threshold",
    min_value=float(lower),
    max_value=float(upper),
    value=float((mu_pos + mu_neg) / 2),
    step=0.01
)

# --- Compute and layout ---
metrics = compute_distributions_and_metrics(auc, prevalence, n, threshold, sigma_pos, sigma_neg)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.image(generate_combined_plot(metrics, threshold))

with col_right:
    st.markdown(f"""
**Prevalence**: {prevalence:.2f}  
**AUC**: {metrics['auc_score']:.2f}  
**Sensitivity (TPR)**: {metrics['TPR']:.2f}  
**Specificity (TNR)**: {metrics['TNR']:.2f}  
**Accuracy**: {metrics['accuracy']:.2f}  
**Balanced Accuracy**: {metrics['balanced_accuracy']:.2f}  
**F1 Score**: {metrics['f1_score']:.2f}  
""")

    st.table(pd.DataFrame({
        "Predicted Negative": [round(metrics["TN"]), round(metrics["FN"])],
        "Predicted Positive": [round(metrics["FP"]), round(metrics["TP"])]
    }, index=["Actual Negative", "Actual Positive"]))
