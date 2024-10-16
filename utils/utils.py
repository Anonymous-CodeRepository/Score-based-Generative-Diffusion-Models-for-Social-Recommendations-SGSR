import torch
import numpy as np
import random
from utils.ema import ExponentialMovingAverage
from model.sde_lib import VPSDE, VESDE

import sys

def set_seed(random_seed):
    torch.manual_seed(random_seed) # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed) # gpu
        torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed) # numpy
    random.seed(random_seed) # random and transforms
    torch.backends.cudnn.deterministic=True # cudnn deterministic calculation   
    print(f'Random seed: ' + str(random_seed))
    
    

def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num) # user_num: number of record ([low,high),size)
    allPos = dataset.allPos
    
    S = []
    
    for i, user in enumerate(users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
    
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.n_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
    return np.array(S)

def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')
    
    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size')

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)
            
def load_sde(sde_type, config):
    beta_min = config['beta_min']
    beta_max = config['beta_max']
    num_scales = config['num_scale']
    
    if sde_type == 'VP':
        sde = VPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
        sampling_eps = 1e-3
    elif sde_type == 'VE':
        sde = VESDE(sigma_min=beta_min, sigma_max=beta_max, N=num_scales)
        sampling_eps = 1e-3
    # elif sde_type == 'subVP':
    #     sde = subVPSDE(beta_min=beta_min, beta_max=beta_max, N=num_scales)
    else:
        raise NotImplementedError(f"SDE class {sde_type} not yet supported.")
    return sde, sampling_eps

def load_ema(model, decay=0.999):
    ema = ExponentialMovingAverage(model.parameters(), decay=decay)
    return ema

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def test_one_batch(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}