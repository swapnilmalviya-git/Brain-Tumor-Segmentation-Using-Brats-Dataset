import math
# from pytorch_msssim import ssim
import cv2
import torch
import numpy as np



# ################################################
# '''Metrics and conversion'''
# #################################################


def mse(x, y):
    return ((x - y)**2).mean()


################################################
'''metrics for Brain Tumor segmentation'''
#################################################


def multi_class_dice(y_true, y_pred, axis=(0,1,2), epsilon=0.00000001):
    
    dice_numerator1 = 2* torch.sum(y_true[:,1,:,:,:]*y_pred[:,1,:,:,:]).cuda() + epsilon
    dice_denominator1 = torch.sum(y_true[:,1,:,:,:]).cuda() + torch.sum(y_pred[:,1,:,:,:]).cuda() + epsilon
    dice_coeff1 = dice_numerator1/dice_denominator1
    
    dice_numerator2 = 2* torch.sum(y_true[:,2,:,:,:]*y_pred[:,2,:,:,:]).cuda() + epsilon
    dice_denominator2 = torch.sum(y_true[:,2,:,:,:]).cuda() + torch.sum(y_pred[:,2,:,:,:]).cuda() + epsilon
    dice_coeff2 = dice_numerator2/dice_denominator2
    
    dice_numerator3 = 2* torch.sum(y_true[:,3,:,:,:]*y_pred[0,3,:,:,:]).cuda() + epsilon
    dice_denominator3 = torch.sum(y_true[:,3,:,:,:]).cuda() + torch.sum(y_pred[:,3,:,:,:]).cuda() + epsilon
    dice_coeff3 = dice_numerator3/dice_denominator3
    
    dice_coeff = (dice_coeff1 + dice_coeff2 + dice_coeff3)/3
    
    return dice_coeff,dice_coeff1,dice_coeff2,dice_coeff3


def multi_class_dice2(y_true, y_pred, axis=(0,1,2), epsilon=0.00000001):
    
    dice_numerator0 = 2* torch.sum(y_true[:,0,:,:]*y_pred[:,0,:,:]).cuda() + epsilon
    dice_denominator0 = torch.sum(y_true[:,0,:,:]).cuda() + torch.sum(y_pred[:,0,:,:]).cuda() + epsilon
    dice_coeff0 = dice_numerator0/dice_denominator0
    
    dice_numerator1 = 2* torch.sum(y_true[:,1,:,:]*y_pred[:,1,:,:]).cuda() + epsilon
    dice_denominator1 = torch.sum(y_true[:,1,:,:]).cuda() + torch.sum(y_pred[:,1,:,:]).cuda() + epsilon
    dice_coeff1 = dice_numerator1/dice_denominator1
    
    dice_numerator2 = 2* torch.sum(y_true[:,2,:,:]*y_pred[:,2,:,:]).cuda() + epsilon
    dice_denominator2 = torch.sum(y_true[:,2,:,:]).cuda() + torch.sum(y_pred[:,2,:,:]).cuda() + epsilon
    dice_coeff2 = dice_numerator2/dice_denominator2
    
    dice_numerator3 = 2* torch.sum(y_true[:,3,:,:]*y_pred[0,3,:,:]).cuda() + epsilon
    dice_denominator3 = torch.sum(y_true[:,3,:,:]).cuda() + torch.sum(y_pred[:,3,:,:]).cuda() + epsilon
    dice_coeff3 = dice_numerator3/dice_denominator3
    
    dice_coeff = (0.1*dice_coeff0 + 0.3*dice_coeff1 + 0.3*dice_coeff2 + 0.3*dice_coeff3)
    
    return dice_coeff,dice_coeff0,dice_coeff1,dice_coeff2,dice_coeff3

def multi_class_dice3(y_true, y_pred, axis=(0,1,2), epsilon=0.00000001):
    
    dice_numerator1 = 2* torch.sum(y_true[:,1,:,:]*y_pred[:,1,:,:]).cuda() + epsilon
    dice_denominator1 = torch.sum(y_true[:,1,:,:]).cuda() + torch.sum(y_pred[:,1,:,:]).cuda() + epsilon
    dice_coeff1 = dice_numerator1/dice_denominator1
    
    dice_numerator2 = 2* torch.sum(y_true[:,2,:,:]*y_pred[:,2,:,:]).cuda() + epsilon
    dice_denominator2 = torch.sum(y_true[:,2,:,:]).cuda() + torch.sum(y_pred[:,2,:,:]).cuda() + epsilon
    dice_coeff2 = dice_numerator2/dice_denominator2
    
    dice_numerator3 = 2* torch.sum(y_true[:,3,:,:]*y_pred[0,3,:,:]).cuda() + epsilon
    dice_denominator3 = torch.sum(y_true[:,3,:,:]).cuda() + torch.sum(y_pred[:,3,:,:]).cuda() + epsilon
    dice_coeff3 = dice_numerator3/dice_denominator3
    
    dice_coeff = (dice_coeff1 + dice_coeff2 + dice_coeff3)/3
    
    return dice_coeff,dice_coeff1,dice_coeff2,dice_coeff3

def dice_loss(truth,predicted):
    truth = torch.flatten(truth)
    predicted =torch.flatten(predicted)
    inter = sum(truth*predicted)
    return (2.*inter + 1)/ (sum(truth) + sum(predicted) + 1) #smooth=1

def sensitivity(truth,predicted):
    truth = torch.argmax(truth[-1]).type(torch.FloatTensor)
    predicted = torch.argmax(predicted[-1]).type(torch.FloatTensor)
    truth_f = torch.flatten(truth)
    predicted_f = torch.flatten(predicted)
    # true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    true_pred = torch.sum(torch.round(torch.clip(truth_f*predicted_f,0,1)))
    # possible_positives = K.sum(K.round(K.clip(y_true_f, 0, 1)))
    possible_pred = torch.sum(torch.round(torch.clip(truth_f,0,1)))
    # return true_positives / (possible_positives + K.epsilon())
    return true_pred/(possible_pred + 0.05)

def specificity(truth,predicted):
    truth = torch.argmax(truth[-1]).type(torch.FloatTensor)
    predicted = torch.argmax(predicted[-1]).type(torch.FloatTensor)
    truth_f = torch.flatten(truth)
    predicted_f = torch.flatten(predicted)
    # true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    true_false = torch.sum(torch.round(torch.clip((1-truth_f) * (1-predicted_f), 0, 1)))
    # possible_positives = K.sum(K.round(K.clip(y_true_f, 0, 1)))
    possible_false = torch.sum(torch.round(torch.clip((1-truth_f), 0, 1)))
    # return true_positives / (possible_positives + K.epsilon())
    return true_false / (possible_false + 0.05)

from scipy.spatial.distance import directed_hausdorff
def hausdorff_distance(y_true, predicted):
    y_true = torch.argmax(y_true[-1]).type(torch.FloatTensor) # (?, ?)
    # y_pred = K.cast(K.argmax(y_pred, axis=-1), "float32") # (?, 50176)
    predicted = torch.argmax(predicted[-1]).type(torch.FloatTensor)
    y_true_f = torch.flatten(y_true) # (?,)
    predicted_f = torch.flatten(predicted) # (?,)

    hd, _, _ = directed_hausdorff(y_true_f, predicted_f)
    return hd




