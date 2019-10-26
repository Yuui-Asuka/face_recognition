from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import source.data_prepare
import matplotlib.pyplot as plt

def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy,best_threshold = source.data_prepare.calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = source.data_prepare.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-2, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far,best_threshold

def get_paths(test_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
           # path0 = add_extension(os.path.join(test_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
           # path1 = add_extension(os.path.join(test_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            path0 = add_extension(os.path.join(test_dir, pair[0], pair[0] + ' ' + '(' + (pair[1]) + ')'))
            path1 = add_extension(os.path.join(test_dir, pair[0], pair[0] + ' ' + '(' + (pair[2]) + ')'))
            issame = True
        elif len(pair) == 4:
           # path0 = add_extension(os.path.join(test_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
           # path1 = add_extension(os.path.join(test_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            path0 = add_extension(os.path.join(test_dir, pair[0], pair[0] + ' ' + '(' + (pair[1]) + ')'))
            path1 = add_extension(os.path.join(test_dir, pair[2], pair[2] + ' ' + '(' + (pair[3]) + ')'))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list
  
def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

