
# import torch
# import torchvision
# from torch.utils.data import DataLoader

import torch
from torch import Tensor
from sklearn.metrics import confusion_matrix





def GDiceLoss(y_true, y_pred, ignore_background=True, weighed=False):
    """
    Generalized Dice Loss
    
    Input:
        y_true: ground truth mask [batch_size, num_classes, height, width] (num_classe could also be 1)
        y_pred: predicted mask [batch_size, num_classes, height, width]
    Output:
        dice_score: generalized dice loss
    """
    assert y_true.shape == y_pred.shape, f"y_true and y_pred must have the same shape {y_true.shape} vs {y_pred.shape}"
    # print("Shape of loss vector", y_true.shape) #16,1,96,96
    
    tp = torch.sum(y_true * y_pred, dim=(0,2,3))
    fp = torch.sum(y_true*(1-y_pred),dim=(0,2,3))
    fn = torch.sum((1-y_true)*y_pred,dim=(0,2,3))
    
    nominator = 2*tp + 1e-05
    denominator = 2*tp + fp + fn + 1e-9
    
    if ignore_background and y_true.shape[1] > 1:
        #print("Ignoring background class")
        dice_score = 1 -(nominator / (denominator+1e-9))[1:]
    else:
        #print("Considering all classes")
        dice_score = 1 -(nominator / (denominator+1e-9))
    
    # ------------------------ # Weighted Dice Loss # ------------------------ #
    # TODO: add weights for each class (It depends on the dataset and number of classes)
    if weighed:
        weighed_dice_score = torch.tensor([0.1, 0.9]).cuda()
        dice_score = torch.mean(weighed_dice_score * dice_score)
    # ------------------------# -------------------- # ------------------------ #
    else:
        dice_score = torch.mean(dice_score) # Average over all classes (or all classes except background)
        
    return dice_score

# ------------------------ # -------------------------------- # ------------------------ #



def avg_class_acc(y_true, y_pred):
    """Average class accuracy
    :param y_true: Ground truth target values.
    :param y_pred: Predicted targets returned by a model.
    :return: Average class accuracy. True negatives. False positives. False negatives. True positives.
    """
    tn, fp, fn, tp = binary_conf_mat_values(y_true, y_pred)
    P_acc = tp / (tp + fn)
    N_acc = tn / (tn + fp)
    avg_acc = (P_acc + N_acc) / 2
    return avg_acc, tn, fp, fn, tp


def binary_conf_mat_values(y_true, y_pred):
    """Binary confusion matrix
    Computes confusion matrix for a binary classification problem.
    :param y_true: Ground truth target values.
    :param y_pred: Predicted targets returned by a model.
    :return: True negatives. False positives. False negatives. True positives.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    conf_mat_values = confusion_matrix(y_true_f, y_pred_f).ravel().tolist()

    # the scikit-learn-confusion-matrix function returns only one number if all of the predicted targets are only
    # true negatives or true positives. The following code adds zeros to the other fields from the confusion matrix.
    if len(conf_mat_values) == 1:
        for i in range(3):
            conf_mat_values.append(0)
        # to check if the one number returned by the scikit-learn-confusion-matrix is value for true positives or
        # true negatives.
        if y_true.sum() / len(y_true_f) == 1:
            conf_mat_values = conf_mat_values[::-1]
    tn, fp, fn, tp = conf_mat_values
    return tn, fp, fn, tp