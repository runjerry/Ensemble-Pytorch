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
