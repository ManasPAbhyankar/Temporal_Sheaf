from models.sheaf_models import LocalConcatSheafLearnerVariant,LocalConcatSheafLearner
from torch import nn
import torch
import torch_sparse
import torch.nn.functional as F
from models import laplacian_builders as lb
import lib.laplace as lap
import math

def sparse_kron(input: torch.Tensor, other: torch.Tensor,device = 'cpu'):
    
    ''' Kronecker product for two sparse matrices
    Code taken from https://github.com/pytorch/pytorch/issues/134069 . All credit to https://github.com/SnoodMcGuinness '''
    assert input.ndim == other.ndim
    input_indices = input.indices()
    other_indices = other.indices()
    # print("Inside Sparse kron ################",input_indices.get_device())
    # print("Inside Sparse kron ################",other_indices.get_device())
    # print("Inside Sparse kron ################",torch.tensor(other.shape).get_device())

    input_indices_expanded = input_indices.expand(other_indices.shape[1], *input_indices.shape).T * (torch.tensor(other.shape).to(device)).reshape(1,-1,1)
    other_indices_expanded = other_indices.expand(input_indices.shape[1], *other_indices.shape)
    # print("Inside Sparse kron ################",input.values().get_device())
    # print("Inside Sparse kron ################",other.values().get_device())
    new_indices = torch.permute(input_indices_expanded + other_indices_expanded, (1,0,2)).reshape(input.ndim,-1)

    new_values = torch.kron(input.values(), other.values())

    if new_indices.ndim == 1:
        new_indices = new_indices.reshape([input.ndim, 0])

    new_shape = [n * m for n, m in zip(input.shape, other.shape)]

    return torch.sparse_coo_tensor(new_indices, new_values, new_shape, dtype=input.dtype, device=input.device).coalesce().to(device)






class SheafTransformerAttentionLayer(nn.Module): 
    def __init__(self,d,graph_size,hidden_dim):
        super(SheafTransformerAttentionLayer,self).__init__()
        #ADD OTHER PARAMETERS HERE Like layer for Vq , Vk and Vv
        # ADD Sheaf Learner Too
        #self.sheaf_learner = LocalConcatSheafLearner(self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act)
        self.d = d
        self.graph_size = graph_size
        self.hidden_dim = hidden_dim
        self.q_linear = nn.Linear(self.hidden_dim,self.hidden_dim ,bias = False )
        self.k_linear = nn.Linear(self.hidden_dim,self.hidden_dim ,bias = False )
        self.v_linear = nn.Linear(self.hidden_dim,self.hidden_dim ,bias = False )
        
        
        
    def forward(self,x,edge_index,sheaf_lap):
        
        #Apply softmax only on incident edges in outer class
            
        #Add attention matrices
        
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        row,col = edge_index
        q_row = torch.index_select(q,dim = 0,index = row)
        k_col = torch.index_select(k,dim = 0,index = col)
        
        # Now , we have created a vector for q and ks for all the nodes in edge index. Since we now need an attention matrix (likely sparse),
        # We will use torch sparse spmm/spspmm probably
        
        
        
        attention_matrix = torch.sum(q_row * k_col,dim  = -1)/math.sqrt(self.hidden_dim)
        # Sparse form of (attention matrix*Adjacency matrix) (Since only q and k of neighbourhood nodes have been dot-producted),
        # Problem remains of softmax now ......
        #can convert to sparse coo tensor and apply softmax

        attention_matrix = torch.sparse_coo_tensor(edge_index,attention_matrix,(x.size(0),x.size(0)))
        attention_matrix = torch.sparse.softmax(attention_matrix,dim = 1)
        # Verify if this is correct
        # with indices contained in edge index (think later if this is wrong or not)

        #where is normalization ???? Done !

        #x = x.reshape(self.graph_size,-1)

        v = v.reshape(self.graph_size*self.d,-1)
        #print(self.graph_size,x.size(0))

        
        
        '''Problem now is to get the adjacency matrix from laplacian ....... 
        .....can use torch_sparse coalesce with op = add (Look at build_sheaf_diffusion_matrix in laplace.py)'''
        
        #adjacency_matrix = lap.build_sheaf_difussion_matrix(x.size(0),self.d,self.edge_index,maps)
        #Cannot use this , use colesce function.....
        #To create adjacency matrix

        # identity_matrix = torch.eye(x.size(0)*self.d).to_sparse()
        # id_idx,id_val = identity_matrix.indices(),identity_matrix.values()
        id_idx = torch.arange(0,x.size(0)*self.d).view(1,-1).tile(2,1)
        id_val = torch.ones((x.size(0)*self.d,))

        # print(id_idx)
        # print(L[0].size)

        adjacency_matrix_idx= torch.cat((id_idx,sheaf_lap[0]),dim = 1)
        adjacency_matrix_val = torch.cat((id_val,-sheaf_lap[1]),dim = 0)

        adjacency_matrix_idx,adjacency_matrix_val = torch_sparse.coalesce(adjacency_matrix_idx,adjacency_matrix_val,self.graph_size*self.d,self.graph_size*self.d,op = 'add')

        
        #right_term = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), v)
        right_term = torch_sparse.spmm(adjacency_matrix_idx, adjacency_matrix_val, self.graph_size*self.d, self.graph_size*self.d, v)
        ''' Check that L must be changed to A!!!! '''
        
        '''The issue in following line is that attention matrix is NOT Dense'''
        high_dim_attn = sparse_kron(attention_matrix.coalesce() , torch.ones((self.d,self.d)).to_sparse())
        return torch_sparse.spmm(high_dim_attn.indices(),high_dim_attn.values(),self.graph_size*self.d,self.graph_size*self.d,right_term)
        
        #return attention_matrix.coalesce(),v

class SheafTransformerMultiHeadAttentionLayer(nn.Module): 
    def __init__(self,d,graph_size ,hidden_dim,device='cpu',num_heads=1):
        super(SheafTransformerMultiHeadAttentionLayer,self).__init__()
        #ADD OTHER PARAMETERS HERE Like layer for Vq , Vk and Vv
        # ADD Sheaf Learner Too
        #self.sheaf_learner = LocalConcatSheafLearner(self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act)
        self.d = d
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.graph_size = graph_size
        
        self.q_linear = nn.Linear(self.hidden_dim,self.hidden_dim*self.num_heads ,bias = False )
        self.k_linear = nn.Linear(self.hidden_dim,self.hidden_dim*self.num_heads ,bias = False )
        self.v_linear = nn.Linear(self.hidden_dim,self.hidden_dim*self.num_heads ,bias = False )

        self.final_linear = nn.Linear(self.hidden_dim*self.num_heads ,self.hidden_dim,bias = False )
        self.device = device
        
        
        
    def forward(self,x,edge_index,sheaf_lap):
        
        #Apply softmax only on incident edges in outer class
            
        #Add attention matrices
        
        
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        q = torch.cat(q.split(q.size(1)//self.num_heads,dim = 1),dim = 0)
        k = torch.cat(k.split(k.size(1)//self.num_heads,dim = 1),dim = 0)
        v = torch.cat(v.split(v.size(1)//self.num_heads,dim = 1),dim = 0) # To put the Qs,Ks and Vs of different attention heads one after the other

        arr = torch.arange(self.num_heads)*self.graph_size
        arr = arr.to(self.device)

        arr = arr[:,None,None]

        arr = arr.tile(2,edge_index.size(1))
        arr = torch.cat(list(arr),dim = 1)

        new_edge_index = edge_index.tile(self.num_heads)
        #print(new_edge_index.shape)
        #print(edge_index.shape)


        new_edge_index = new_edge_index + arr #Got indices of all attention matrices in a single matrix (Imagine them as blocks along diagonal of a much larger matrix)
        
        row,col = new_edge_index
        q_row = torch.index_select(q,dim = 0,index = row)
        k_col = torch.index_select(k,dim = 0,index = col)
        attention_matrix = torch.sum(q_row * k_col,dim  = -1)/math.sqrt(self.hidden_dim)#Got sparse attention matrix values here for all heads
        #print(attention_matrix.shape)

    
        
        # Now , we have created a vector for q and ks for all the nodes in edge index. Since we now need an attention matrix (likely sparse),
        # We will use torch sparse spmm/spspmm probably
        

        attention_matrix = torch.sparse_coo_tensor(new_edge_index,attention_matrix,(self.graph_size*self.num_heads,self.graph_size*self.num_heads))
        attention_matrix = torch.sparse.softmax(attention_matrix,dim = 1)
        attention_matrix = attention_matrix.coalesce()
        # Verify if this is correct
        # with indices contained in edge index (think later if this is wrong or not)
        
        #return attention_matrix.coalesce(),v (put sheaf laplacian here as well)

        v = v.reshape(-1,self.hidden_dim//self.d)


        #Getting adjacency matrix from laplacian

        id_idx = torch.arange(0,self.graph_size*self.d).view(1,-1).tile(2,1).to(self.device)
        id_val = torch.ones((self.graph_size*self.d,)).to(self.device)

        # print(id_idx)
        # print(L[0].size)

        adjacency_matrix_idx = torch.cat((id_idx,sheaf_lap[0]),dim = 1)
        adjacency_matrix_val = torch.cat((id_val,-sheaf_lap[1]),dim = 0)

        adjacency_matrix_idx,adjacency_matrix_val = torch_sparse.coalesce(adjacency_matrix_idx,adjacency_matrix_val,self.graph_size*self.d,self.graph_size*self.d,op = 'add')

        adjacency_matrix = torch.sparse_coo_tensor(adjacency_matrix_idx,adjacency_matrix_val,(self.graph_size*self.d,self.graph_size*self.d)).coalesce()

        sparse_identity_num_heads = torch.sparse_coo_tensor(torch.arange(0,self.num_heads).view(1,-1).tile(2,1),torch.ones((self.num_heads,)),(self.num_heads,self.num_heads)).coalesce().to(self.device)
        #print("###############################---",adjacency_matrix.get_device())
        #print("###############################---",sparse_identity_num_heads.get_device())

        kroneckered_sheaflap = sparse_kron(sparse_identity_num_heads,adjacency_matrix,self.device)

        kroneckered_attention_matrix = sparse_kron(attention_matrix,torch.ones(self.d,self.d).to_sparse().to(self.device),self.device)

        #print(kroneckered_attention_matrix.size(0),kroneckered_attention_matrix.size(1))
        #print(kroneckered_sheaflap.size(0),kroneckered_sheaflap.size(1))
        #print(v.size(0),v.size(1))

        x = torch_sparse.spmm(kroneckered_attention_matrix.indices(),kroneckered_attention_matrix.values(),
                            kroneckered_attention_matrix.size(0),kroneckered_attention_matrix.size(1),
                            torch_sparse.spmm(kroneckered_sheaflap.indices(),kroneckered_sheaflap.values(),
                            kroneckered_sheaflap.size(0),kroneckered_sheaflap.size(1),v))
        
        #x = x.reshape((self.graph_size,-1)) #Rehsaping looks wrong : Corresponding xs for each node must be concatenated

        x = torch.cat(x.split(x.size(0)//self.num_heads,dim = 0),dim = 1)
        x = x.reshape(self.graph_size,-1)
        x = self.final_linear(x)

        x = x.reshape((self.graph_size*self.d,-1))

        #x = F.

        return x

        #Check if this works correctly
        











class SheafTransformerAttentionLayer(nn.Module): 
    def __init__(self,d,graph_size,hidden_dim):
        super(SheafTransformerAttentionLayer,self).__init__()
        #ADD OTHER PARAMETERS HERE Like layer for Vq , Vk and Vv
        # ADD Sheaf Learner Too
        #self.sheaf_learner = LocalConcatSheafLearner(self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act)
        self.d = d
        self.graph_size = graph_size
        self.hidden_dim = hidden_dim
        self.q_linear = nn.Linear(self.hidden_dim,self.hidden_dim ,bias = False )
        self.k_linear = nn.Linear(self.hidden_dim,self.hidden_dim ,bias = False )
        self.v_linear = nn.Linear(self.hidden_dim,self.hidden_dim ,bias = False )
        
        
        
    def forward(self,x,edge_index,sheaf_lap):
        
        #Apply softmax only on incident edges in outer class
            
        #Add attention matrices
        
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        row,col = edge_index
        q_row = torch.index_select(q,dim = 0,index = row)
        k_col = torch.index_select(k,dim = 0,index = col)
        
        # Now , we have created a vector for q and ks for all the nodes in edge index. Since we now need an attention matrix (likely sparse),
        # We will use torch sparse spmm/spspmm probably
        
        
        
        attention_matrix = torch.sum(q_row * k_col,dim  = -1)/math.sqrt(self.hidden_dim)
        # Sparse form of (attention matrix*Adjacency matrix) (Since only q and k of neighbourhood nodes have been dot-producted),
        # Problem remains of softmax now ......
        #can convert to sparse coo tensor and apply softmax

        attention_matrix = torch.sparse_coo_tensor(edge_index,attention_matrix,(x.size(0),x.size(0)))
        attention_matrix = torch.sparse.softmax(attention_matrix,dim = 1)
        # Verify if this is correct
        # with indices contained in edge index (think later if this is wrong or not)

        #where is normalization ???? Done !

        #x = x.reshape(self.graph_size,-1)

        v = v.reshape(self.graph_size*self.d,-1)
        #print(self.graph_size,x.size(0))

        
        
        '''Problem now is to get the adjacency matrix from laplacian ....... 
        .....can use torch_sparse coalesce with op = add (Look at build_sheaf_diffusion_matrix in laplace.py)'''
        
        #adjacency_matrix = lap.build_sheaf_difussion_matrix(x.size(0),self.d,self.edge_index,maps)
        #Cannot use this , use colesce function.....
        #To create adjacency matrix

        # identity_matrix = torch.eye(x.size(0)*self.d).to_sparse()
        # id_idx,id_val = identity_matrix.indices(),identity_matrix.values()
        id_idx = torch.arange(0,x.size(0)*self.d).view(1,-1).tile(2,1)
        id_val = torch.ones((x.size(0)*self.d,))

        # print(id_idx)
        # print(L[0].size)

        adjacency_matrix_idx= torch.cat((id_idx,sheaf_lap[0]),dim = 1)
        adjacency_matrix_val = torch.cat((id_val,-sheaf_lap[1]),dim = 0)

        adjacency_matrix_idx,adjacency_matrix_val = torch_sparse.coalesce(adjacency_matrix_idx,adjacency_matrix_val,self.graph_size*self.d,self.graph_size*self.d,op = 'add')

        
        #right_term = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), v)
        right_term = torch_sparse.spmm(adjacency_matrix_idx, adjacency_matrix_val, self.graph_size*self.d, self.graph_size*self.d, v)
        ''' Check that L must be changed to A!!!! '''
        
        '''The issue in following line is that attention matrix is NOT Dense'''
        high_dim_attn = sparse_kron(attention_matrix.coalesce() , torch.ones((self.d,self.d)).to_sparse())
        return torch_sparse.spmm(high_dim_attn.indices(),high_dim_attn.values(),self.graph_size*self.d,self.graph_size*self.d,right_term)
        
        #return attention_matrix.coalesce(),v





class SheafTransformerMultiHeadAttentionLayerV2(nn.Module): 
    def __init__(self,d,graph_size ,hidden_dim,device='cpu',num_heads=1):
        super(SheafTransformerMultiHeadAttentionLayerV2,self).__init__()
        #ADD OTHER PARAMETERS HERE Like layer for Vq , Vk and Vv
        # ADD Sheaf Learner Too
        #self.sheaf_learner = LocalConcatSheafLearner(self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act)
        self.d = d
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.graph_size = graph_size
        
        self.q_linear = nn.Linear(self.hidden_dim,self.hidden_dim*self.num_heads ,bias = False )
        self.k_linear = nn.Linear(self.hidden_dim,self.hidden_dim*self.num_heads ,bias = False )
        self.v_linear = nn.Linear(self.hidden_dim,self.hidden_dim*self.num_heads ,bias = False )

        self.final_linear = nn.Linear(self.hidden_dim*self.num_heads ,self.hidden_dim,bias = False )
        self.device = device
        
        
        
    def forward(self,x,edge_index,sheaf_lap):
        
        #Apply softmax only on incident edges in outer class
            
        #Add attention matrices
        
        
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        q = torch.cat(q.split(q.size(1)//self.num_heads,dim = 1),dim = 0)
        k = torch.cat(k.split(k.size(1)//self.num_heads,dim = 1),dim = 0)
        v = torch.cat(v.split(v.size(1)//self.num_heads,dim = 1),dim = 0) # To put the Qs,Ks and Vs of different attention heads one after the other

        arr = torch.arange(self.num_heads)*self.graph_size
        arr = arr.to(self.device)

        arr = arr[:,None,None]

        arr = arr.tile(2,edge_index.size(1))
        arr = torch.cat(list(arr),dim = 1)

        new_edge_index = edge_index.tile(self.num_heads)
        #print(new_edge_index.shape)
        #print(edge_index.shape)


        new_edge_index = new_edge_index + arr #Got indices of all attention matrices in a single matrix (Imagine them as blocks along diagonal of a much larger matrix)
        
        row,col = new_edge_index
        q_row = torch.index_select(q,dim = 0,index = row)
        k_col = torch.index_select(k,dim = 0,index = col)
        attention_matrix = torch.sum(q_row * k_col,dim  = -1)/math.sqrt(self.hidden_dim)#Got sparse attention matrix values here for all heads
        #print(attention_matrix.shape)

    
        
        # Now , we have created a vector for q and ks for all the nodes in edge index. Since we now need an attention matrix (likely sparse),
        # We will use torch sparse spmm/spspmm probably
        

        attention_matrix = torch.sparse_coo_tensor(new_edge_index,attention_matrix,(self.graph_size*self.num_heads,self.graph_size*self.num_heads))
        attention_matrix = torch.sparse.softmax(attention_matrix,dim = 1)
        attention_matrix = attention_matrix.coalesce()
        # Verify if this is correct
        # with indices contained in edge index (think later if this is wrong or not)
        
        #return attention_matrix.coalesce(),v (put sheaf laplacian here as well)

        v = v.reshape(-1,self.hidden_dim//self.d)


        #Getting adjacency matrix from laplacian

        id_idx = torch.arange(0,self.graph_size*self.d).view(1,-1).tile(2,1).to(self.device)
        id_val = torch.ones((self.graph_size*self.d,)).to(self.device)

        # print(id_idx)
        # print(L[0].size)

        adjacency_matrix_idx = torch.cat((id_idx,sheaf_lap[0]),dim = 1)
        adjacency_matrix_val = torch.cat((id_val,-sheaf_lap[1]),dim = 0)

        adjacency_matrix_idx,adjacency_matrix_val = torch_sparse.coalesce(adjacency_matrix_idx,adjacency_matrix_val,self.graph_size*self.d,self.graph_size*self.d,op = 'add')

        adjacency_matrix = torch.sparse_coo_tensor(adjacency_matrix_idx,adjacency_matrix_val,(self.graph_size*self.d,self.graph_size*self.d)).coalesce()

        sparse_identity_num_heads = torch.sparse_coo_tensor(torch.arange(0,self.num_heads).view(1,-1).tile(2,1),torch.ones((self.num_heads,)),(self.num_heads,self.num_heads)).coalesce().to(self.device)
        #print("###############################---",adjacency_matrix.get_device())
        #print("###############################---",sparse_identity_num_heads.get_device())

        kroneckered_sheaflap = sparse_kron(sparse_identity_num_heads,adjacency_matrix,self.device)

        kroneckered_attention_matrix = sparse_kron(attention_matrix,torch.ones(self.d,self.d).to_sparse().to(self.device),self.device)

        #print(kroneckered_attention_matrix.size(0),kroneckered_attention_matrix.size(1))
        #print(kroneckered_sheaflap.size(0),kroneckered_sheaflap.size(1))
        #print(v.size(0),v.size(1))

        # x = torch_sparse.spmm(kroneckered_attention_matrix.indices(),kroneckered_attention_matrix.values(),
        #                     kroneckered_attention_matrix.size(0),kroneckered_attention_matrix.size(1),
        #                     torch_sparse.spmm(kroneckered_sheaflap.indices(),kroneckered_sheaflap.values(),
        #                     kroneckered_sheaflap.size(0),kroneckered_sheaflap.size(1),v))

        #This is now the dot product of all attention matrices with all the sheaf laplacians..........

        kroneckered_attention_matrix = kroneckered_attention_matrix.to_dense()
        mult_terms = kroneckered_attention_matrix[torch.tensor_split(kroneckered_sheaflap.indices(),2)[0],torch.tensor_split(kroneckered_sheaflap.indices(),2)[1]]

        left_term_values = mult_terms * kroneckered_sheaflap.values()

        x = torch_sparse.spmm(kroneckered_sheaflap.indices(),left_term_values,
                             kroneckered_sheaflap.size(0),kroneckered_sheaflap.size(1),v)[0]


        #print(x.shape,v.shape,x.size(0))
        #x = x[0]
        
        #x = x.reshape((self.graph_size,-1)) #Reshaping looks wrong : Corresponding xs for each node must be concatenated

        x = torch.cat(x.split(x.size(0)//self.num_heads,dim = 0),dim = 1)
        x = x.reshape(self.graph_size,-1)
        x = self.final_linear(x)

        x = x.reshape((self.graph_size*self.d,-1))

        #x = F.

        return x

        #Check if this works correctly





class SheafTransformerMultiHeadAttentionLayerV3(nn.Module): 
    def __init__(self,d,graph_size ,hidden_dim,device='cpu',num_heads=1):
        super(SheafTransformerMultiHeadAttentionLayerV3,self).__init__()

        '''Adding left and eight weights for V calculation...'''
        #ADD OTHER PARAMETERS HERE Like layer for Vq , Vk and Vv
        # ADD Sheaf Learner Too
        #self.sheaf_learner = LocalConcatSheafLearner(self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act)
        self.d = d
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.graph_size = graph_size
        
        self.q_linear = nn.Linear(self.hidden_dim,self.hidden_dim*self.num_heads ,bias = False )
        self.k_linear = nn.Linear(self.hidden_dim,self.hidden_dim*self.num_heads ,bias = False )
        self.v_linear_2 = nn.Linear(self.hidden_dim//self.d,(self.hidden_dim//self.d)*self.num_heads ,bias = False )

        self.v_linear_1 = nn.Parameter(torch.randn(self.num_heads,self.d,self.d))

        self.final_linear = nn.Linear(self.hidden_dim*self.num_heads ,self.hidden_dim,bias = False )
        self.device = device
        
        
        
    def forward(self,x,edge_index,sheaf_lap):
        
        #Apply softmax only on incident edges in outer class
            
        #Add attention matrices
        
        
        q = self.q_linear(x)
        k = self.k_linear(x)
        #v = self.v_linear(x)

        q = torch.cat(q.split(q.size(1)//self.num_heads,dim = 1),dim = 0)
        k = torch.cat(k.split(k.size(1)//self.num_heads,dim = 1),dim = 0)
        #v = torch.cat(v.split(v.size(1)//self.num_heads,dim = 1),dim = 0) # To put the Qs,Ks and Vs of different attention heads one after the other

        arr = torch.arange(self.num_heads)*self.graph_size
        arr = arr.to(self.device)

        arr = arr[:,None,None]

        arr = arr.tile(2,edge_index.size(1))
        arr = torch.cat(list(arr),dim = 1)

        new_edge_index = edge_index.tile(self.num_heads)
        #print(new_edge_index.shape)
        #print(edge_index.shape)


        new_edge_index = new_edge_index + arr #Got indices of all attention matrices in a single matrix (Imagine them as blocks along diagonal of a much larger matrix)
        
        row,col = new_edge_index
        q_row = torch.index_select(q,dim = 0,index = row)
        k_col = torch.index_select(k,dim = 0,index = col)
        attention_matrix = torch.sum(q_row * k_col,dim  = -1)/math.sqrt(self.hidden_dim)#Got sparse attention matrix values here for all heads
        #print(attention_matrix.shape)

    
        
        # Now , we have created a vector for q and ks for all the nodes in edge index. Since we now need an attention matrix (likely sparse),
        # We will use torch sparse spmm/spspmm probably
        

        attention_matrix = torch.sparse_coo_tensor(new_edge_index,attention_matrix,(self.graph_size*self.num_heads,self.graph_size*self.num_heads))
        attention_matrix = torch.sparse.softmax(attention_matrix,dim = 1)
        attention_matrix = attention_matrix.coalesce()
        # Verify if this is correct
        # with indices contained in edge index (think later if this is wrong or not)
        
        #return attention_matrix.coalesce(),v (put sheaf laplacian here as well)



        #Getting the V matrix from Xs

        v_right = self.v_linear_2(x.view(-1,self.hidden_dim//self.d)) #get multi+head values of shape (num_nodes*d,self.hidden_dim*self.num_heads//self.d)


        v_right = torch.cat(torch.split(v_right,self.hidden_dim//self.d,dim = 1),dim = 0).reshape(shape = (self.num_heads,-1,self.hidden_dim//self.d))


        v_right = torch.einsum('bij -> bji',v_right).reshape(shape = (self.num_heads,-1,self.d))

        v = torch.einsum('bij,bjk -> bik',v_right,self.v_linear_1)

        v = v.reshape(shape = (self.num_heads,self.hidden_dim//self.d,-1))

        v = torch.einsum('bij -> bji',v).reshape(shape = (self.num_heads*self.graph_size*self.d,-1))



        







        #v = v.reshape(-1,self.hidden_dim//self.d)


        #Getting adjacency matrix from laplacian

        id_idx = torch.arange(0,self.graph_size*self.d).view(1,-1).tile(2,1).to(self.device)
        id_val = torch.ones((self.graph_size*self.d,)).to(self.device)

        # print(id_idx)
        # print(L[0].size)

        adjacency_matrix_idx = torch.cat((id_idx,sheaf_lap[0]),dim = 1)
        adjacency_matrix_val = torch.cat((id_val,-sheaf_lap[1]),dim = 0)

        adjacency_matrix_idx,adjacency_matrix_val = torch_sparse.coalesce(adjacency_matrix_idx,adjacency_matrix_val,self.graph_size*self.d,self.graph_size*self.d,op = 'add')

        adjacency_matrix = torch.sparse_coo_tensor(adjacency_matrix_idx,adjacency_matrix_val,(self.graph_size*self.d,self.graph_size*self.d)).coalesce()

        sparse_identity_num_heads = torch.sparse_coo_tensor(torch.arange(0,self.num_heads).view(1,-1).tile(2,1),torch.ones((self.num_heads,)),(self.num_heads,self.num_heads)).coalesce().to(self.device)
        #print("###############################---",adjacency_matrix.get_device())
        #print("###############################---",sparse_identity_num_heads.get_device())

        kroneckered_sheaflap = sparse_kron(sparse_identity_num_heads,adjacency_matrix,self.device)

        kroneckered_attention_matrix = sparse_kron(attention_matrix,torch.ones(self.d,self.d).to_sparse().to(self.device),self.device)

        #print(kroneckered_attention_matrix.size(0),kroneckered_attention_matrix.size(1))
        #print(kroneckered_sheaflap.size(0),kroneckered_sheaflap.size(1))
        #print(v.size(0),v.size(1))

        # x = torch_sparse.spmm(kroneckered_attention_matrix.indices(),kroneckered_attention_matrix.values(),
        #                     kroneckered_attention_matrix.size(0),kroneckered_attention_matrix.size(1),
        #                     torch_sparse.spmm(kroneckered_sheaflap.indices(),kroneckered_sheaflap.values(),
        #                     kroneckered_sheaflap.size(0),kroneckered_sheaflap.size(1),v))

        #This is now the dot product of all attention matrices with all the sheaf laplacians..........

        kroneckered_attention_matrix = kroneckered_attention_matrix.to_dense()
        mult_terms = kroneckered_attention_matrix[torch.tensor_split(kroneckered_sheaflap.indices(),2)[0],torch.tensor_split(kroneckered_sheaflap.indices(),2)[1]]

        left_term_values = mult_terms * kroneckered_sheaflap.values()

        x = torch_sparse.spmm(kroneckered_sheaflap.indices(),left_term_values,
                             kroneckered_sheaflap.size(0),kroneckered_sheaflap.size(1),v)[0]


        #print(x.shape,v.shape,x.size(0))
        #x = x[0]
        
        #x = x.reshape((self.graph_size,-1)) #Reshaping looks wrong : Corresponding xs for each node must be concatenated

        x = torch.cat(x.split(x.size(0)//self.num_heads,dim = 0),dim = 1)
        x = x.reshape(self.graph_size,-1)
        x = self.final_linear(x)

        x = x.reshape((self.graph_size*self.d,-1))

        #x = F.

        return x

        #Check if this works correctly




























        



            
            
class SheafTransformerBase(nn.Module):
    """Base class for sheaf Transformer models."""

    def __init__(self, edge_index, args):
        super(SheafTransformerBase, self).__init__()

        assert args['d'] > 0
        self.d = args['d']
        self.edge_index = edge_index
        self.add_lp = args['add_lp'] #What does this do ?
        self.add_hp = args['add_hp'] #What does this do ?

        self.final_d = self.d
        if self.add_hp:
            self.final_d += 1
        if self.add_lp:
            self.final_d += 1

        self.hidden_dim = args['hidden_channels'] * self.final_d
        self.device = args['device']
        self.graph_size = args['graph_size']
        #self.layers = args['layers']
        self.normalised = args['normalised']
        self.deg_normalised = args['deg_normalised']
        self.nonlinear = not args['linear']
        self.input_dropout = args['input_dropout']
        self.dropout = args['dropout']
        #self.left_weights = args['left_weights']
        #self.right_weights = args['right_weights']
        self.sparse_learner = args['sparse_learner']
        self.use_act = args['use_act']
        self.input_dim = args['input_dim']
        self.hidden_channels = args['hidden_channels']
        self.output_dim = args['output_dim']
        self.layers = args['layers']
        
        self.num_attention_layers = args['num_attention_layers']
        self.num_heads = args['num_heads']
        
        self.sheaf_act = args['sheaf_act']
        #self.second_linear = args['second_linear']
        self.orth_trans = args['orth']
        self.use_edge_weights = args['edge_weights']
        #self.t = args['max_t']
        #self.time_range = torch.tensor([0.0, self.t], device=self.device)
        self.laplacian_builder = None

    def update_edge_index(self, edge_index):
        assert edge_index.max() <= self.graph_size
        self.edge_index = edge_index
        self.laplacian_builder = self.laplacian_builder.create_with_new_edge_index(edge_index)

    def grouped_parameters(self):
        sheaf_learners, others = [], []
        for name, param in self.named_parameters():
            if "sheaf_learner" in name:
                sheaf_learners.append(param)
            else:
                others.append(param)
        assert len(sheaf_learners) > 0
        assert len(sheaf_learners) + len(others) == len(list(self.parameters()))
        return sheaf_learners, others





class GeneralSheafTransformer(SheafTransformerBase):

    def __init__(self, edge_index, args):
        super(GeneralSheafTransformer, self).__init__(edge_index, args)
        assert args['d'] > 1

        #self.lin_right_weights = nn.ModuleList()
        #self.lin_left_weights = nn.ModuleList()

        # if self.right_weights:
        #     for i in range(self.layers):
        #         self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
        #         nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        # if self.left_weights:
        #     for i in range(self.layers):
        #         self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
        #         nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.attention_layers = nn.ModuleList()

        #num_attention_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(self.num_attention_layers):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
            #self.attention_layers.append(SheafTransformerAttentionLayer(d = self.d,graph_size = self.graph_size,hidden_dim = self.hidden_dim))
            # self.attention_layers.append(SheafTransformerMultiHeadAttentionLayer(d = self.d,graph_size = self.graph_size,hidden_dim = self.hidden_dim,device = self.device,num_heads = self.num_heads))
            self.attention_layers.append(SheafTransformerMultiHeadAttentionLayerV3(d = self.d,graph_size = self.graph_size,hidden_dim = self.hidden_dim,device = self.device,num_heads = self.num_heads))
            
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        # self.epsilons = nn.ParameterList()
        # for i in range(self.layers):
        #     self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        # if self.second_linear:
        #     self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    # def left_right_linear(self, x, left, right):
    #     if self.left_weights:
    #         x = x.t().reshape(-1, self.final_d)
    #         x = left(x)
    #         x = x.reshape(-1, self.graph_size * self.final_d).t()

    #     if self.right_weights:
    #         x = right(x)

    #     return x

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # if self.second_linear:
        #     x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.num_attention_layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)
                #print(trans_maps.shape)
                #print(maps.shape)

            x = F.dropout(x, p=self.dropout, training=self.training)

            #x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])
            
            
            
            '''#Here do the attention matrix and V thing, will take some time to formulate in terms of pytorch functions
            # Use the adjacency matrix rather than the diagonal'''
            
            
            '''Getting (sparse) attention matrix and V values'''

            x = x.reshape(-1,self.hidden_dim)
            
            x = self.attention_layers[layer](x, self.edge_index, L)

            
            #x = x.to_dense()
            '''scipy sparse kron can be used BUT backprop is gone'''
            '''
            Take this link for sparse kronecker : https://github.com/pytorch/pytorch/issues/134069 (Look at last post)
            '''
            
            
            '''Residual term has to be created ........... Check the equation'''
            
            
            
            
            
            #x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)
            
            # coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))

            x = x0 + x #Adding residual term
            #x = x0

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1) # Return Representations (without activation) Or return softmaxed values
    
    
    
    
    
    
    '''
    
    TODO : Complete the General sheaf transformer end - to end , and verify that it runs correctly (Working correctly for one epoch)
    TODO: Add MultiHead Attention and verify its working(Done)
    TODO : Put hyperparameter tuning, and test it on a concrete dataset
    TODO : Adapt the General Sheaf transformer to Diagonal and Discrete Bundles by chaging relevant lines
    
    '''