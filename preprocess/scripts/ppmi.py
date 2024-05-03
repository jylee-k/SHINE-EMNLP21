import numpy as np
import math
from scipy.sparse import coo_matrix

def PPMI(inputs, mapping, window_size, sparse):
    """
    Calculate the Pointwise Pointwise Mutual Information (PPMI) between words in the given inputs.
    
    Args:
        inputs: list of strings, the input sentences
        mapping: dict, a mapping of words to indices
        window_size: int, the size of the window for word co-occurrence
        sparse: bool, flag to indicate whether to use a sparse matrix representation
    
    Returns:
        PMI_adj: numpy.ndarray or scipy.sparse.coo_matrix, the PMI matrix representing the mutual information between words
    
    """
    W_ij = np.zeros([len(mapping), len(mapping)], dtype=np.float64)
    W_i = np.zeros([len(mapping)], dtype=np.float64)
    W_count = 0
    for sent in inputs:
        word_list = sent.split(' ')
        if len(word_list) - window_size < 0:
            window_num = 1
        else:
            window_num = len(word_list) - window_size + 1
        for i in range(window_num):
            W_count += 1
            context = list(set(word_list[i:i + window_size]))
            while '' in context:
                context.remove('')
            for j in range(len(context)):
                W_i[mapping[context[j]]] += 1
                for k in range(j + 1, len(context)):
                    W_ij[mapping[context[j]], mapping[context[k]]] += 1
                    W_ij[mapping[context[k]], mapping[context[j]]] += 1
    if sparse:
        rows = []
        columns = []
        data = []
        for i in range(len(mapping)):
            rows.append(i)
            columns.append(i)
            data.append(1)
            tmp = [ele for ele in np.nonzero(W_ij[i])[0] if ele > i]
            for j in tmp:
                value = math.log(W_ij[i, j] * W_count / W_i[i] / W_i[j])
                if value > 0:
                    rows.append(i)
                    columns.append(j)
                    data.append(value)
                    rows.append(j)
                    columns.append(i)
                    data.append(value)
        PMI_adj = coo_matrix((data, (rows, columns)), shape=(len(mapping), len(mapping)))
    else:
        PMI_adj = np.zeros([len(mapping), len(mapping)], dtype=np.float64)
        for i in range(len(mapping)):
            PMI_adj[i, i] = 1  
            tmp = [ele for ele in np.nonzero(W_ij[i])[0] if ele > i] 
            # for j in range(i + 1, len(mapping)):
            for j in tmp:
                pmi = math.log(W_ij[i, j] * W_count / W_i[i] / W_i[j])
                if pmi > 0:
                    PMI_adj[i, j] = pmi
                    PMI_adj[j, i] = pmi
    return PMI_adj