import scipy.optimize
import numpy as np
import torch
import torch.nn.functional as F

# Object detector util: match a set of detected bounding boxes and a set of ground-truth bounding boxes
# https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4

def bbox_iou(boxA, boxB):
  # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
  # ^^ corrected.
    
  # Determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  interW = max(0, xB - xA)
  interH = max(0, yB - yA)

  # Correction: reject non-overlapping boxes
  if interW <=0 or interH <=0 :
    return -1.0

  interArea = interW * interH
  boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
  iou = interArea / float(boxAArea + boxBArea - interArea)
  return iou


def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.5):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
      The number of bboxes, N1 and N2, need not be the same.
    
    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i,:], bbox_pred[j,:])

    if n_pred > n_true:
      # there are more predictions than ground-truth - add dummy rows
      diff = n_pred - n_true
      iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((diff, n_pred), MIN_IOU)), 
                                  axis=0)

    if n_true > n_pred:
      # more ground-truth than predictions - add dummy columns
      diff = n_true - n_pred
      iou_matrix = np.concatenate( (iou_matrix, 
                                    np.full((n_true, diff), MIN_IOU)), 
                                  axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred<n_pred
    idx_pred_actual = idxs_pred[sel_pred] 
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label


def DetLoss(bbox_gt, bbox_pred, conf_pred, IOU_THRESH=0.1):
    """
    Calculate the combined loss for bounding boxes and confidence scores using PyTorch,
    including BCE for confidence loss and accounting for false negatives.
    
    Parameters:
    - bbox_gt: PyTorch tensor of ground truth bounding boxes, shape (N, 4)
    - bbox_pred: PyTorch tensor of predicted bounding boxes, shape (5, 4)
    - conf_pred: PyTorch tensor of predicted confidence scores, shape (5,)
    
    Returns:
    - Total loss, bounding box loss, confidence loss, and false negative penalty
    """
    idx_gt, idx_pred, ious, _ = match_bboxes(bbox_gt.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy(), IOU_THRESH=IOU_THRESH)
    # print("Matched GT:", idx_gt)
    # print("Matched Pre:", idx_pred)
    # print("IOU:", ious)
    
    # Smooth L1 Loss for matched bounding boxes
    if idx_gt.size == 0:
       bbox_loss = torch.tensor(1.0, dtype=torch.float, device=bbox_gt.device, requires_grad=True)
    else:
      matched_gt_boxes = bbox_gt[idx_gt]
      matched_pred_boxes = bbox_pred[idx_pred]
      bbox_loss = F.smooth_l1_loss(100*matched_pred_boxes, 100*matched_gt_boxes)
    
    # Prepare targets for BCE loss
    target_conf = torch.zeros_like(conf_pred)
    target_conf[idx_pred] = 1  # Mark matched predictions with high confidence
    conf_loss = F.binary_cross_entropy(conf_pred, target_conf)
    
    # False Negative Penalty: Additional penalty for each GT box not matched
    fn_penalty = 0.1  # Adjust based on model performance and dataset
    fn_loss = (len(bbox_gt) - len(idx_gt)) * fn_penalty
    fn_loss = torch.tensor(fn_loss, dtype=torch.float, device=bbox_gt.device, requires_grad=True)
    
    total_loss = bbox_loss + conf_loss + fn_loss
    return total_loss, bbox_loss, conf_loss, fn_loss


def DetLossBatch(bbox_gt, bbox_pred, conf_pred, IOU_THRESH=0.3):
    """
    Calculate the combined loss for bounding boxes and confidence scores for a batch using PyTorch.
    
    Parameters:
    - bbox_gt: PyTorch tensor of ground truth bounding boxes, shape (B, N, 4)
    - bbox_pred: PyTorch tensor of predicted bounding boxes, shape (B, 5, 4)
    - conf_pred: PyTorch tensor of predicted confidence scores, shape (B, 5)
    """
    batch_size = bbox_gt.shape[0]
    total_loss_sum = 0
    bbox_loss_sum = 0
    conf_loss_sum = 0
    fn_loss_sum = 0
    
    for i in range(batch_size):
        # Extract the bounding boxes and confidence scores for the current item in the batch
        bbox_gt_item = bbox_gt[i]
        bbox_pred_item = bbox_pred[i]
        conf_pred_item = conf_pred[i]
        
        # Calculate the loss for the current item
        total_loss, bbox_loss, conf_loss, fn_loss = DetLoss(bbox_gt_item, bbox_pred_item, conf_pred_item, IOU_THRESH)
        total_loss_sum += total_loss
        bbox_loss_sum += bbox_loss
        conf_loss_sum += conf_loss
        fn_loss_sum += fn_loss
    
    return total_loss_sum/batch_size, bbox_loss_sum/batch_size, conf_loss_sum/batch_size, fn_loss_sum/batch_size