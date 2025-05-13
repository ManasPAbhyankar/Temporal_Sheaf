import numpy as np

def input_conv(x):
    # x is embedding of all nodes in the graph... of shape (num_nodes,embedding_dim)

    lfc = (1/x.shape[0])*(np.ones((x.shape[0],x.shape[0]))@x)
    hfc = np.eye(x.shape[0]) - lfc 

    return np.linalg.norm(hfc,ord = 2)/np.linalg.norm(lfc,ord = 2)


def angle_conv(x):

    normed_x = x/np.linalg.norm(x,axis = 1,keepdims = True)

    tril_sims = np.tril(normed_x@normed_x.T)

    return (2/(x.shape[0]*(x.shape[0]-1)))*np.sum(tril_sims)

def rank_collapse(x):

    singular_vals = np.linalg.svd(x,compute_uv = False)

    normed_vals = singular_vals/np.sum(singular_vals)

    return np.exp(np.sum(-normed_vals*np.log(normed_vals)))
