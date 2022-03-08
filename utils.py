'''
Modified Date: 2022/03/08
Author: Gi-Luen Huang
mail: come880412@gmail.com
'''

import torch
import cv2
import numpy as np
from skimage.transform import resize


def blur(img):
    k_size = 11
    bl = cv2.GaussianBlur(img,(k_size,k_size),0)
    return torch.FloatTensor(bl)

def kldiv(s_map, gt):
    assert s_map.size() == gt.size()
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)
    
    assert expand_s_map.size() == s_map.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)
    
    assert expand_gt.size() == gt.size()

    s_map = s_map/(expand_s_map*1.0)
    gt = gt / (expand_gt*1.0)

    s_map = s_map.view(batch_size, -1)
    gt = gt.view(batch_size, -1)

    eps = 2.2204e-16
    result = gt * torch.log(eps + gt/(s_map + eps))
    # print(torch.log(eps + gt/(s_map + eps))   )
    return torch.mean(torch.sum(result, 1))

def cc(s_map, gt):
    assert s_map.size() == gt.size()
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    mean_s_map = torch.mean(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_s_map = torch.std(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    mean_gt = torch.mean(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_gt = torch.std(gt.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    s_map = (s_map - mean_s_map) / std_s_map
    gt = (gt - mean_gt) / std_gt

    ab = torch.sum((s_map * gt).view(batch_size, -1), 1)
    aa = torch.sum((s_map * s_map).view(batch_size, -1), 1)
    bb = torch.sum((gt * gt).view(batch_size, -1), 1)

    return torch.mean(ab / (torch.sqrt(aa*bb)))

def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)
    
    min_s_map = torch.min(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)
    max_s_map = torch.max(s_map.view(batch_size, -1), 1)[0].view(batch_size, 1, 1).expand(batch_size, w, h)

    norm_s_map = (s_map - min_s_map)/(max_s_map-min_s_map*1.0)
    return norm_s_map

def similarity(s_map, gt):
    ''' For single image metric
        Size of Image - WxH or 1xWxH
        gt is ground truth saliency map
    '''
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)

    s_map = normalize_map(s_map)
    gt = normalize_map(gt)
    
    sum_s_map = torch.sum(s_map.view(batch_size, -1), 1)
    expand_s_map = sum_s_map.view(batch_size, 1, 1).expand(batch_size, w, h)
    
    assert expand_s_map.size() == s_map.size()

    sum_gt = torch.sum(gt.view(batch_size, -1), 1)
    expand_gt = sum_gt.view(batch_size, 1, 1).expand(batch_size, w, h)

    s_map = s_map/(expand_s_map*1.0)
    gt = gt / (expand_gt*1.0)

    s_map = s_map.view(batch_size, -1)
    gt = gt.view(batch_size, -1)
    return torch.mean(torch.sum(torch.min(s_map, gt), 1))

def nss(s_map, gt):
    if s_map.size() != gt.size():
        s_map = s_map.cpu().squeeze(0).numpy()
        s_map = torch.FloatTensor(cv2.resize(s_map, (gt.size(2), gt.size(1)))).unsqueeze(0)
        s_map = s_map.cuda(1)
        gt = gt.cuda(1)
    # print(s_map.size(), gt.size())
    assert s_map.size()==gt.size()
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)
    mean_s_map = torch.mean(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_s_map = torch.std(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    eps = 2.2204e-16
    s_map = (s_map - mean_s_map) / (std_s_map + eps)

    s_map = torch.sum((s_map * gt).view(batch_size, -1), 1)
    count = torch.sum(gt.contiguous().view(batch_size, -1), 1)
    return torch.mean(s_map / count)

def auc_judd(saliencyMap, fixationMap, jitter=True, toPlot=False, normalize=False):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)
    # jitter=True will add tiny non-zero random constant to all map locations to ensure
    #       ROC can be calculated robustly (to avoid uniform region)
    # if toPlot=True, displays ROC curve

    # If there are no fixations to predict, return NaN
    if saliencyMap.size() != fixationMap.size():
        saliencyMap = saliencyMap.cpu().squeeze(0).numpy()
        saliencyMap = torch.FloatTensor(cv2.resize(saliencyMap, (fixationMap.size(2), fixationMap.size(1)))).unsqueeze(0)
        # saliencyMap = saliencyMap.cuda(1)
        # fixationMap = fixationMap.cuda(1)
    if len(saliencyMap.size())==3:
        saliencyMap = saliencyMap[0,:,:]
        fixationMap = fixationMap[0,:,:]
    saliencyMap = saliencyMap.numpy()
    fixationMap = fixationMap.numpy()
    if normalize:
        saliencyMap = normalize_map(saliencyMap)

    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        return score

    # make the saliencyMap the size of the image of fixationMap

    if not np.shape(saliencyMap) == np.shape(fixationMap):
        from scipy.misc import imresize
        saliencyMap = imresize(saliencyMap, np.shape(fixationMap))

    # jitter saliency maps that come from saliency models that have a lot of zero values.
    # If the saliency map is made with a Gaussian then it does not need to be jittered as
    # the values are varied and there is not a large patch of the same value. In fact
    # jittering breaks the ordering in the small values!
    if jitter:
        # jitter the saliency map slightly to distrupt ties of the same numbers
        saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7

    # normalize saliency map
    saliencyMap = (saliencyMap - saliencyMap.min()) \
                  / (saliencyMap.max() - saliencyMap.min())

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        score = float('nan')
        return score

    S = saliencyMap.flatten()
    F = fixationMap.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    allthreshes = sorted(Sth, reverse=True)  # sort sal map values, to sweep through values
    tp = np.zeros((Nfixations + 2))
    fp = np.zeros((Nfixations + 2))
    tp[0], tp[-1] = 0, 1
    fp[0], fp[-1] = 0, 1

    for i in range(Nfixations):
        thresh = allthreshes[i]
        aboveth = (S >= thresh).sum()  # total number of sal map values above threshold
        tp[i + 1] = float(i + 1) / Nfixations  # ratio sal map values at fixation locations
        # above threshold
        fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)  # ratio other sal map values
        # above threshold

    score = np.trapz(tp, x=fp)
    allthreshes = np.insert(allthreshes, 0, 0)
    allthreshes = np.append(allthreshes, 1)

    if toPlot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.matshow(saliencyMap, cmap='gray')
        ax.set_title('SaliencyMap with fixations to be predicted')
        [y, x] = np.nonzero(fixationMap)
        s = np.shape(saliencyMap)
        plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
        plt.plot(x, y, 'ro')

        ax = fig.add_subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        ax.set_title('Area under ROC curve: ' + str(score))
        plt.axis((0, 1, 0, 1))
        plt.show()

    return score

def sauc(s_map, gt, other_map, n_splits=100, stepsize=0.1):
    # pdb.set_trace()
    # If there are no fixations to predict, return NaN
    if np.sum(gt) == 0:
        print('no gt')
        return None

    # normalize saliency map
    s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)

    # gt = np.where(gt > 80, 1, 0)
    # other_map = np.where(other_map > 80, 1, 0)

    if s_map.shape != gt.shape:
        s_map = resize(s_map, gt.shape)

    if other_map.shape != gt.shape:
        other_map = resize(other_map, gt.shape)

    S = s_map.flatten()
    F = gt.flatten()
    Oth = other_map.flatten()

    Sth = S[F > 200]  # sal map values at fixation locations #568
    # Sth = S[F == 255]  # sal map values at fixation locations #568

    # Sth = S[F > 0]
    Nfixations = len(Sth)
    # print(Nfixations)

    # for each fixation, sample Nsplits values from the sal map at locations
    # specified by other_map

    Oth[ np.where(F > 200) ] = 0
    # Oth[ np.where(F > 255) ] = 0

    ind = np.where(Oth > 0)[0]  # find fixation locations on other images #621
    # ind = np.where(Oth > 200)[0]

    Nfixations_oth = min(Nfixations, len(ind))
    randfix = np.full((Nfixations_oth, n_splits), np.nan)
    # randfix = np.full((len(ind), n_splits), np.nan)

    for i in range(n_splits):
        # randomize choice of fixation locations
        randind = np.random.permutation(ind.copy()) #621
        # sal map values at random fixation locations of other random images
        randfix[:, i] = S[randind[:Nfixations_oth]]
        # randfix[:, i] = S[randind[:]]


    # calculate AUC per random split (set of random locations)
    auc = np.full(n_splits, np.nan)
    # pdb.set_trace()
    for s in range(n_splits):

        curfix = randfix[:, s]

        allthreshes = np.flip(np.arange(0, max(np.max(Sth), np.max(curfix)), stepsize))
        tp = np.zeros(len(allthreshes) + 2)
        fp = np.zeros(len(allthreshes) + 2)
        tp[-1] = 1
        fp[-1] = 1

        for i in range(len(allthreshes)):
            thresh = allthreshes[i]
            tp[i + 1] = np.sum(Sth >= thresh) / Nfixations
            fp[i + 1] = np.sum(curfix >= thresh) / Nfixations_oth

        auc[s] = np.trapz(np.array(tp), np.array(fp))
    # return (tp, fp, randind, allthreshes)
    return np.mean(auc)
