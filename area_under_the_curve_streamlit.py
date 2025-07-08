import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from sklearn.metrics import roc_auc_score
import streamlit as st


matplotlib.use('Agg')  # Required for Streamlit compatibility


# --- Data Generation ---
@st.cache_data
def generate_distributions(auc, prevalence=0.5, n=1000):
    d_prime = np.sqrt(2) * norm.ppf(auc)
    mu_pos = d_prime
    mu_neg = 0
    sigma = 1

    n_pos = int(n * prevalence)
    n_neg = int(n * (1 - prevalence))

    pos = np.random.normal(loc=mu_pos, scale=sigma, size=n_pos)
    neg = np.random.normal(loc=mu_neg, scale=sigma, size=n_neg)

    return pos, neg

# --- Plotting Function ---
def plot_roc_and_distributions(pos, neg, threshold, prevalence):
    from sklearn.metrics import confusion_matrix
    import matplotlib.gridspec as gridspec

    n_pos = len(pos)
    n_neg = len(neg)
    n_total = n_pos + n_neg
    p_pos = n_pos / n_total
    p_neg = n_neg / n_total

    # KDEs
    kde_pos = gaussian_kde(pos)
    kde_neg = gaussian_kde(neg)

    x_vals = np.linspace(min(pos.min(), neg.min()) - 1, max(pos.max(), neg.max()) + 1, 1000)
    y_pos = kde_pos(x_vals) * p_pos
    y_neg = kde_neg(x_vals) * p_neg

    y_true = np.array([1] * n_pos + [0] * n_neg)
    y_scores = np.concatenate([pos, neg])
    preds = y_scores >= threshold

    TP = np.sum((preds == 1) & (y_true == 1))
    FP = np.sum((preds == 1) & (y_true == 0))
    FN = np.sum((preds == 0) & (y_true == 1))
    TN = np.sum((preds == 0) & (y_true == 0))

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    auc_score = roc_auc_score(y_true, y_scores)

    # ROC curve points
    thresholds = np.linspace(min(y_scores), max(y_scores), 200)
    tpr_list = []
    fpr_list = []
    for t in thresholds:
        preds_t = y_scores >= t
        TP_t = np.sum((preds_t == 1) & (y_true == 1))
        FP_t = np.sum((preds_t == 1) & (y_true == 0))
        FN_t = np.sum((preds_t == 0) & (y_true == 1))
        TN_t = np.sum((preds_t == 0) & (y_true == 0))
        tpr = TP_t / (TP_t + FN_t) if (TP_t + FN_t) > 0 else 0
        fpr = FP_t / (FP_t + TN_t) if (FP_t + TN_t) > 0 else 0
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Create figure and layout
    fig = plt.figure(figsize=(20, 7), dpi=600)
    gs = gridspec.GridSpec(3, 5, figure=fig)

    # Row 1, Col 1: Information
    ax_info = fig.add_subplot(gs[0, 0])
    ax_info.axis('off')
    info_text = (
        f"Prevalence:        {prevalence:.2f}\n"
        f"AUC:               {auc_score:.3f}\n"
        f"Sensitivity:       {TPR:.3f}\n"
        f"Specificity:       {specificity:.3f}\n"
    )
    ax_info.text(0, 1, info_text, fontsize=14, verticalalignment='top', family='monospace')
    ax_info.set_title("Model Information")

    # Row 2, Col 1: Confusion Matrix
    ax_cm = fig.add_subplot(gs[1:2, 0])
    ax_cm.axis('off')
    cm = confusion_matrix(y_true, preds)
    cm_labels = [["TN", "FP"], ["FN", "TP"]]
    cell_text = [[f"{cm_labels[i][j]} = {cm[i][j]}" for j in range(2)] for i in range(2)]
    table = ax_cm.table(cellText=cell_text,
                        rowLabels=["True Neg", "True Pos"],
                        colLabels=["Pred Neg", "Pred Pos"],
                        cellLoc='center',
                        loc='upper center',
                        bbox=[0, 0.15, 1, 0.8])
    table.scale(1, 2)
    ax_cm.set_title("Confusion Matrix")

    # KDE plot across cols 1–2 (both rows)
    ax_kde = fig.add_subplot(gs[:, 1:3])
    ax_kde.plot(x_vals, y_pos, label='Positive', color='#c0392b')
    ax_kde.plot(x_vals, y_neg, label='Negative', color='#2980b9')
    ax_kde.fill_between(x_vals, 0, y_pos, where=(x_vals >= threshold), color='red', alpha=0.3, label='True Positives')
    ax_kde.fill_between(x_vals, 0, y_neg, where=(x_vals >= threshold), color='blue', alpha=0.3, label='False Positives')
    ax_kde.axvline(threshold, color='black', linestyle='--', label='Threshold')
    ax_kde.set_title("Score Distributions")
    ax_kde.set_xlabel("Score")
    ax_kde.set_ylabel("Density")
    #ax_kde.legend(loc='center right', bbox_to_anchor=(-0.25, 0.5))
    ax_kde.legend(loc='center right')

    # ROC curve across cols 3–4 (both rows)
    ax_roc = fig.add_subplot(gs[:, 3:5])
    # Fill under the ROC curve
    ax_roc.fill_between(fpr_list, tpr_list, step='mid', alpha=0.2, color='#aeb6bf', label='AUC Area')
    # Plot ROC curve
    ax_roc.plot(fpr_list, tpr_list, label='ROC Curve', color='black')
    # Add diagonal line (random classifier)
    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance (AUC=0.5)')
    # Current threshold point
    ax_roc.scatter(FPR, TPR, color='red', s=80, label='Current Threshold')
    # Axes and labels
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()

    plt.tight_layout()
    #st.pyplot(fig)
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")  # DPI is respected here
    buf.seek(0)
    st.image(buf)



# --- Streamlit UI ---
st.title("Interactive ROC Curve Visualizer")
st.set_page_config(layout="wide")

# Sidebar sliders
auc = st.sidebar.slider("Desired AUC", min_value=0.5, max_value=0.99, value=0.85, step=0.01)
prevalence = st.sidebar.slider("Prevalence (positive class)", min_value=0.01, max_value=0.99, value=0.5, step=0.01)
n = st.sidebar.slider("Number of Samples", min_value=100, max_value=10000, value=5000, step=100)


# Add your credentials "at the bottom"
st.sidebar.markdown("---")

st.sidebar.markdown("""
# About the Author
### Nils R. Winter, PhD 

https://nilsrwinter.com
 
https://github.com/NilsWinter 
""")

# Generate data (now cached)
pos, neg = generate_distributions(auc=auc, prevalence=prevalence, n=n)

# Get score range from generated data
all_scores = np.concatenate([pos, neg])
min_score = float(all_scores.min())
max_score = float(all_scores.max())

# Threshold slider
threshold = st.slider(
    "Decision Threshold",
    min_value=min_score,
    max_value=max_score,
    value=np.median(all_scores),
    step=0.01
)
# Plot
plot_roc_and_distributions(pos, neg, threshold, prevalence)
