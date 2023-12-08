import torch
import numpy as np


def dice_round(preds, trues):
    preds = preds.float()
    return soft_dice_loss(preds, trues)


def soft_dice_loss(outputs, targets, per_image=False): # bs, h, w / bs, h, w
    eps = 1e-6
    batch_size = outputs.size()[0]
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()

    return loss



def dice_round_fn(predicted, ground_truth, score_threshold=0.5, area_threshold=0):

    mask = predicted > score_threshold
    if mask.sum() < area_threshold:
        mask = torch.zeros_like(mask)
    return 1 - dice_round(mask, ground_truth).item()


def search_thresholds(eval_list, thr_list, area_list):
    best_score = 0
    best_thr = -1
    best_area = -1

    for thr in thr_list:
        for area in area_list:
            score_list = []
            for probas, labels in eval_list:
                score = dice_round_fn(probas, labels, thr, area)
                score_list.append(score)
            final_score = np.mean(score_list)
            if final_score > best_score:
                best_score = final_score
                best_thr = thr
                best_area = area
    return best_thr, best_area, best_score