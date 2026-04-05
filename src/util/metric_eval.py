import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve
from typing import List, Dict, Union
import torch.nn.functional as F


class ECELoss(torch.nn.Module):
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        self.bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = self.bin_boundaries[:-1]
        self.bin_uppers = self.bin_boundaries[1:]

    def forward(self, accuracies: torch.Tensor, confidences: torch.Tensor):
        ece = torch.zeros(1, device=confidences.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = ((confidences >= bin_lower) & (confidences < bin_upper))
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += prop_in_bin * torch.abs(avg_confidence_in_bin - accuracy_in_bin)
        return ece
    

def compute_nll(logits: torch.Tensor, labels: torch.Tensor, top_versus_all: bool):
    if top_versus_all:
        probs = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.max(probs, dim=1)
        binary_labels = (labels == top_indices).float().to(logits.device)
        return F.binary_cross_entropy(top_probs, binary_labels)
    else:
        return F.cross_entropy(logits, labels)


def draw_search_curve(probs, logits, labels, accuracies, best_temperature, save_path, top_versus_all):
    # plot the NLL and ECE curves with respect to temperature
    ece_criterion = ECELoss()
    t_vals = np.linspace(0.1, 5.0, 50)
    nll_vals, ece_vals = [], []
    with torch.no_grad():
        for t in tqdm(t_vals, desc="Plotting temperature curves"):
            scaled_logits = logits / t
            probs = torch.softmax(scaled_logits, dim=1).max(dim=1)[0]
            nll_vals.append(compute_nll(scaled_logits, labels, top_versus_all).item())
            ece_vals.append(ece_criterion(accuracies, probs).item())
    plt.figure(figsize=(10, 5))
    plt.plot(t_vals, nll_vals, label='NLL', color='blue')
    plt.plot(t_vals, ece_vals, label='ECE', color='orange')
    plt.axvline(x=best_temperature.item(), color='red', linestyle='--', label='Optimal Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Loss')
    plt.title('NLL and ECE vs Temperature')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def search_temperature(logits: torch.Tensor, labels: torch.Tensor, 
                       device: str, lr: float=0.01, max_iter: int=50, 
                       save_path: Union[None, str]=None,
                       top_versus_all: bool=False):
    # Search for the optimal temperature
    ece_criterion = ECELoss()
    probs, preds = torch.softmax(logits, dim=1).max(dim=1)
    accuracies = (preds == labels).float()
    
    before_temperature_nll = compute_nll(logits, labels, top_versus_all)
    before_temperature_ece = ece_criterion(accuracies, probs) # softmax probility as confidence
    print(f"Before temperature: NLL: {before_temperature_nll.item():.4f}, ECE: {before_temperature_ece.item():.4f}")

    temperature = torch.tensor(1.0, requires_grad=True, device=device)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    def eval():
        optimizer.zero_grad()
        loss = compute_nll(logits / temperature, labels, top_versus_all)
        loss.backward()
        return loss
    optimizer.step(eval)

    probs, preds = torch.softmax(logits / temperature, dim=1).max(dim=1)
    after_temperature_nll = compute_nll(logits / temperature, labels, top_versus_all)
    after_temperature_ece = ece_criterion(accuracies, probs)
    print(f"After temperature: NLL: {after_temperature_nll.item():.4f}, ECE: {after_temperature_ece.item():.4f}")
    print(f"Optimal temperature: {temperature.item():.4f}")

    if save_path is not None:
        draw_search_curve(probs, logits, labels, accuracies, temperature, save_path, top_versus_all)

    return temperature.item()


def compute_confidence(logits: torch.Tensor, conf_type: str):
    # Compute the confidence scores
    probs = torch.softmax(logits, dim=1)
    if conf_type == "max_logit":
        return torch.max(logits, dim=1)[0]
    elif conf_type == "max_prob":
        return torch.max(probs, dim=1)[0]
    elif conf_type == "energy":
        return torch.logsumexp(logits, dim=1)
    elif conf_type == "entropy":
        return torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    else:
        raise ValueError(f"Unknown confidence type: {conf_type}")
    # num_class = logits.size(1)
    # kl_divs = torch.sum(probs * (torch.log(probs + 1e-10) - torch.log(1.0 / num_class)), dim=1).cpu().numpy()


def draw_reliability_diagram(accuracies: np.ndarray, 
                             confidences: np.ndarray, 
                             n_bins: int=10, save_dir: Union[None, str]=None):
    # Draw the reliability diagram
    plt.rcParams.update({'font.size': 20})

    bin_size = 1.0 / n_bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies, bin_confidences = np.zeros(n_bins), np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    for i in range(n_bins):
        selected = np.where(indices == i + 1)[0]
        if len(selected) > 0:
            bin_accuracies[i] = accuracies[selected].mean()
            bin_confidences[i] = confidences[selected].mean()
            bin_counts[i] = len(selected)
    
    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    def reliability_digram_subplot(draw_ece=True, draw_bin_importance="alpha",
                                   title="Reliability Digram",
                                   xlabel="Confidence", ylabel="Accuracy"):
        positions = bins[:-1] + bin_size / 2
        widths = bin_size
        alphas = 0.3
        min_count, max_count = np.min(bin_counts), np.max(bin_counts)
        normalized_counts = (bin_counts - min_count) / (max_count - min_count + 1e-10)
        if draw_bin_importance == "alpha":
            alphas = 0.2 + 0.8 * normalized_counts
        elif draw_bin_importance == "width":
            widths = 0.1 * bin_size + 0.9 * bin_size * normalized_counts
        colors = np.zeros((n_bins, 4))
        colors[:, 0] = 240 / 255
        colors[:, 1] = 60 / 255
        colors[:, 2] = 60 / 255
        colors[:, 3] = alphas
        plt.figure(figsize=(10, 8))
        gap_plt = plt.bar(positions, gaps, bottom=np.minimum(bin_accuracies, bin_confidences),
                         width=widths, color="#ffbcbb", edgecolor=colors, linewidth=1, label="Gap", hatch="//")
        acc_plt = plt.bar(positions, bin_accuracies, width=widths, edgecolor="blue",
                        color="#2c00ff", linewidth=3, label="Accuracy")
        plt.plot([0, 1], [0, 1], ls="dotted", lw=3, c="gray")
        if draw_ece:
            plt.text(0.98, 0.02, f"ECE={ece*100:.2f}", 
                     color="black", ha="right", va="bottom", 
                    fontsize=26, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(handles=[gap_plt, acc_plt])

        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "reliability_diagram.pdf"), dpi=300, bbox_inches='tight')
        plt.close()

    def confidence_histogram_subplot(draw_averages=True, title="Confidence Histogram",
                                    xlabel="Confidence", ylabel="Count"):
        positions = bins[:-1] + bin_size / 2
        plt.figure(figsize=(10, 8))
        plt.bar(positions, bin_counts, width=bin_size*0.9)
        plt.xlim(0, 1)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if draw_averages:
            acc_plt = plt.axvline(x=avg_acc, ls="solid", lw=3, c="black", label="Accuracy")
            conf_plt = plt.axvline(x=avg_conf, ls="dotted", lw=3, c="#444", label="Confidence")
            plt.legend(handles=[acc_plt, conf_plt])
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "confidence_histogram.pdf"), dpi=300, bbox_inches='tight')
        plt.close()
    
    reliability_digram_subplot()
    confidence_histogram_subplot()


def tpr_fpr(y_prob: np.ndarray, y_true: np.ndarray, thres):
    y_pred = (y_prob < thres).astype(int) # 1 for ood, 0 for id, higher y_prob for higher id confidence 
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr, fpr


def roc_auc_score(y_prob: np.ndarray, y_true: np.ndarray, n_split: int=1000, plot_path=None):
    # thres_min, thres_max = y_prob.min(), y_prob.max()
    # thres_list = np.linspace(thres_min, thres_max, n_split)
    # tpr_list, fpr_list = [], []
    # for thres in thres_list:
    #     tpr, fpr = tpr_fpr(y_prob, y_true, thres)
    #     tpr_list.append(tpr)
    #     fpr_list.append(fpr)
    fpr_list, tpr_list, _ = roc_curve(y_true, -y_prob) # positive (ood) = y_score > threshold
    roc_auc = auc(fpr_list, tpr_list)
    if plot_path is not None:
        plt.figure()
        plt.plot(fpr_list, tpr_list, label=f"ROC AUC: {roc_auc:.4f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(plot_path)
        plt.close()
    return roc_auc


def plot_confidence(confidences: np.ndarray, y_true: np.ndarray, plot_path=None):
    plt.figure(figsize=(10, 8))
    id_idx = np.where(y_true == 0)[0]
    ood_idx = np.where(y_true == 1)[0]
    plt.scatter(id_idx, confidences[id_idx], label="In-distribution", alpha=0.5, color="blue")
    plt.scatter(ood_idx, confidences[ood_idx], label="Out-of-distribution", alpha=0.5, color="red")
    plt.xlabel("Sample Index")
    plt.ylabel("Confidence")
    plt.title("Per-sample Confidence")
    plt.legend()
    if plot_path is not None:
        plt.savefig(plot_path)
    plt.close()


class AUCLoss(torch.nn.Module):
    def __init__(self):
        super(AUCLoss, self).__init__()

    def forward(self, logits, labels, conf_type):
        probs = torch.softmax(logits, dim=1)
        if conf_type == "max_logit":
            max_logits = torch.max(logits, dim=1)[0]
            return -roc_auc_score(max_logits.cpu().numpy(), labels.cpu().numpy())
        elif conf_type == "max_prob":
            max_probs = torch.max(probs, dim=1)[0]
            return -roc_auc_score(max_probs.cpu().numpy(), labels.cpu().numpy())
        elif conf_type == "energy":
            energies = torch.logsumexp(logits, dim=1)
            return -roc_auc_score(energies.cpu().numpy(), labels.cpu().numpy())
        elif conf_type == "entropy":
            entropies = torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            return -roc_auc_score(entropies.cpu().numpy(), labels.cpu().numpy())
        else:
            raise ValueError(f"Unknown confidence type: {conf_type}")


def search_auc_temperature(logits, labels, save_dir, conf_type="max_prob", lr=0.01, max_iter=50):
    # Search for the optimal temperature
    auc_criterion = AUCLoss() # auc loss is not differentiable
    auc_before = auc_criterion(logits, labels, conf_type)
    print(f"Before temperature: AUC = {auc_before:.4f}")

    def get_best_temperature(conf_type):
        least_auc_score = float("inf")
        best_temperature = 0.1
        auc_vals, t_vals = [], np.linspace(0.1, 5.0, 100)
        # auc is not differentiable, so we use grid search
        for t in tqdm(t_vals, desc="Searching temperature"):
            scaled_logits = logits / t
            auc_score = auc_criterion(scaled_logits, labels, conf_type)
            auc_vals.append(auc_score)
            if auc_score < least_auc_score:
                least_auc_score = auc_score
                best_temperature = t
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(10, 5))
        plt.plot(t_vals, auc_vals, label='AUC', color='blue')
        plt.axvline(x=best_temperature, color='red', linestyle='--', label='Optimal Temperature')
        plt.xlabel('Temperature')
        plt.ylabel('AUC')
        plt.title('AUC vs Temperature')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"auc_temperature_{conf_type}.pdf"))
        plt.close()
        return best_temperature

    best_temperature = get_best_temperature(conf_type)
    auc_after = auc_criterion(logits / best_temperature, labels, conf_type)
    print(f"Best temperature for {conf_type}: {best_temperature:.4f}")
    print(f"After temperature: AUC = {auc_after:.4f}")

    return best_temperature


def draw_confusion_matrix(cm, idx2label, save_path=None):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(10, 8))
    class_names = [idx2label[i] for i in range(len(idx2label))]
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def compute_fpr95(confidences, labels):
    """
    计算 FPR95: 当 TPR = 95% 时的 FPR
    Args:
        confidences: 置信度分数，ID样本应该有更高的置信度
        labels: 0 for ID, 1 for OOD
    Returns:
        fpr95: FPR at 95% TPR
    """
    from sklearn.metrics import roc_curve
    
    # 对于OOD检测，我们需要将ID样本的置信度取负值
    # 因为sklearn的roc_curve假设正类（OOD）有更高的分数
    scores = -confidences  # ID样本置信度高，OOD样本置信度低
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # 找到TPR >= 0.95的最小FPR
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 1.0  # 如果无法达到95% TPR，返回最差情况
    
    fpr95 = fpr[idx[0]]
    return fpr95

def compute_auroc(confidences, labels):
    """
    计算 AUROC
    Args:
        confidences: 置信度分数，ID样本应该有更高的置信度  
        labels: 0 for ID, 1 for OOD
    Returns:
        auroc: Area Under ROC Curve
    """
    from sklearn.metrics import roc_auc_score
    
    # 对于OOD检测，我们需要将ID样本的置信度取负值
    scores = -confidences
    return roc_auc_score(labels, scores)