import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_erode(img):
    if len(img.shape)==4:
        p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1,p2)
    elif len(img.shape)==5:
        p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
        p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
        p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    if len(img.shape)==4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    elif len(img.shape)==5:
        return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_):
    img1  =  soft_open(img)
    skel  =  F.relu(img-img1)
    for j in range(iter_):
        img  =  soft_erode(img)
        img1  =  soft_open(img)
        delta  =  F.relu(img-img1)
        skel  =  skel +  F.relu(delta-skel*delta)
    return skel

class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+self.smooth)/(torch.sum(skel_pred[:,1:,...])+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+self.smooth)/(torch.sum(skel_true[:,1:,...])+self.smooth)    
        
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        
        return cl_dice


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred), dim=(0,2,3))
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true, dim=(0,2,3) ) + torch.sum(y_pred, dim=(0,2,3)) + smooth)
    return (1. - coeff)


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth = 1.e-5):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, y_pred, y_true, ignore_background=False):
        dice = soft_dice(y_true, y_pred)
        
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        
        tprec = (torch.sum(torch.multiply(skel_pred, y_true), dim=(0,2,3))+self.smooth)/(torch.sum(skel_pred, dim=(0,2,3))+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred), dim=(0,2,3))+self.smooth)/(torch.sum(skel_true, dim=(0,2,3))+self.smooth)    
        
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        
        return (1.0-self.alpha)*dice+self.alpha*cl_dice
    
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