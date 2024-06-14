import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy as entropy_fn
from sklearn.metrics import roc_auc_score as auroc


def auc_score(known, unknown):
    """ Computes the AUROC for the given predictions on `known` data
        and `unknown` data.
    """
    y_true = np.array([0] * len(known) + [1] * len(unknown))
    y_score = np.concatenate([known, unknown])
    auc_score = auroc(y_true, y_score)
    return auc_score


def entropy_prob(probs, dim=1):
    p = probs
    eps = 1e-12
    logp = torch.log(p + eps)
    plogp = p * logp
    entropy = -torch.sum(plogp, dim=dim)
    return entropy


def uncertainty(outputs): 
    """ outputs (torch.tensor): class probabilities, 
        in practice these are given by a softmax operation
        * Soft voting averages the probabilties across the ensemble
            dimension, and then takes the maximal predicted class
            Taking the entropy of the averaged probabilities does not 
            yield a valid probability distribution, but in practice its ok
    """
    # Soft Voting (entropy and var in confidence)
    if outputs.shape[0] > 1:
        preds_soft = outputs.mean(0)  # [data, dim]
    else:
        preds_soft = outputs[0, :, :]
    entropy = entropy_fn(preds_soft.T.cpu().numpy()) # [data]
    
    preds_hard = outputs.var(0).cpu()  # [data, dim]
    variance = preds_hard.sum(1).numpy()  # [data]
    return (entropy, variance)


    uncertainties = uncertainty(outputs)
    entropy, variance = uncertainties
    uncertainties2 = uncertainty(outputs2)
    entropy2, variance2 = uncertainties2
    auc_entropy = auc_score(entropy, entropy2)
    auc_variance = auc_score(variance, variance2)


def compute_ece(confidences, predictions, labels, num_bins=15):
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # confidences, predictions = np.max(softmax_logits, -1), np.argmax(softmax_logits, -1)
    if isinstance(confidences, torch.Tensor):
        confidences = confidences.cpu().numpy()
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
    else:
        confidences = np.array(confidences)
        predictions = np.array(predictions)
        labels = np.array(labels)
    
    # print(predictions.shape, labels.shape)
    accuracies = predictions == labels
    # print(accuracies)

    ece = 0.0
    avg_confs_in_bins = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            delta = avg_confidence_in_bin - accuracy_in_bin
            avg_confs_in_bins.append(delta)
            ece += np.abs(delta) * prop_in_bin
        else:
            avg_confs_in_bins.append(None)

    return ece
