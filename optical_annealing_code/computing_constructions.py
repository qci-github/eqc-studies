import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import scipy.linalg as linalg_d
import time
import scipy.optimize as opt
import scipy.special as special
import copy
import warnings
import QO_trunc_functions as QOtrunc # base module for truncated quantum optics
import density_matrix_extension as dme # module for handling density matrix calculations

def coherent_driving_time_series(gamma_nl,alpha_inj,calculate_func=lambda x: copy.copy(x),rho_init=5,eta_pump_loss=0,t_max=5*np.pi/np.sqrt(2),nt=200): # function for calcualting the effect of single-mode driving for different times
    '''
        gamma_nl is the rate at which the system couples to the non-linear driving
        alpha_inj is the strength at which new photons are injected linearly
        calculate_func is a function which operates on the intermediate density matrix, defaults to an identity function which just returns the whole matrix
        rho_init can either be supplied as a number, in which case it is the maximum number of photons allowed in the initial mode, or an array, in which case it is taken as the initial density matrix (without the pump mode tensored in yet)
        eta_pump_loss is the single-photon loss rate in the pump mode, defaults to 0
        t_max is the maximum time for which evolution is to be applied, or the only time to be used if nt=1
        nt is the number of evenly-spaced time steps at which values will be returned, if set to 1, than the function will return only the value at t_max
    '''
    if type(rho_init)==int: # if we are starting in vacuum and the mode size has been supplied
        n_trunc=copy.copy(rho_init)
        rho_init=np.zeros([n_trunc+1,n_trunc+1])
        rho_init[0,0]=1 # vacuum density matrix
        rho_init=dme.matrix_rho_to_vector_rho(rho_init)
    else:
        rho_init=dme.convert_rho_to_vector_idem(rho_init) # ensure it is in vector form
        n_trunc=int(np.sqrt(rho_init.shape[0]))-1 # infer the size of the mode from the size of the initial density matrix
    if nt>1: # if more than one time step
        t_step=t_max/(nt-1) # the size of a single time step
    else: # otherwise just step the whole time at once
        t_step=t_max
    (gen_mat,traceout_mat_rho_vec,expand_mat_rho_vec)=dme.Zeno_blockade_displacement_supOp(gamma_nl,alpha_inj,n_trunc=n_trunc,is_coherent=True,eta_pump_loss=eta_pump_loss,return_gen=True) # construct necessary generator matrix
    if nt>1: # if running for more than one time
        time_step_supOp=linalg_d.expm(t_step*gen_mat) # perform matrix exponentiation to build the superoperator which steps the time
        rho_t=expand_mat_rho_vec@rho_init # expand to cover the full Hilbert space
        func_return_list=[] # empty list to return intermediate values
        for i_t in range(nt): # loop through time steps
            func_return_list=func_return_list+[calculate_func(rho_t)] # calculate values to return
            rho_t=time_step_supOp@rho_t # apply time-stepping superoperator
        return func_return_list
    else:
        rho_t=linalg.expm_multiply(t_step*gen_mat,expand_mat_rho_vec@rho_init) # apply time-stepping superoperator
        return calculate_func(rho_t)
    
def tpa_driving_time_series(gamma_loss,alpha_inj,calculate_func=lambda x: copy.copy(x),rho_init=5,t_max=5*np.pi/np.sqrt(2),nt=200): # function for calcualting the effect of single-mode driving for different times
    '''
        gamma_loss is the rate at which the system couples to the loss
        alpha_inj is the strength at which new photons are injected linearly
        calculate_func is a function which operates on the intermediate density matrix, defaults to an identity function which just returns the whole matrix
        rho_init can either be supplied as a number, in which case it is the maximum number of photons allowed in the initial mode, or an array, in which case it is taken as the initial density matrix (without the pump mode tensored in yet)
        t_max is the maximum time for which evolution is to be applied, or the only time to be used if nt=1
        nt is the number of evenly-spaced time steps at which values will be returned, if set to 1, than the function will return only the value at t_max
    '''
    if type(rho_init)==int: # if we are starting in vacuum and the mode size has been supplied
        n_trunc=copy.copy(rho_init)
        rho_init=np.zeros([n_trunc+1,n_trunc+1])
        rho_init[0,0]=1 # vacuum density matrix
        rho_init=dme.matrix_rho_to_vector_rho(rho_init)
    else:
        rho_init=dme.convert_rho_to_vector_idem(rho_init) # ensure it is in vector form
        n_trunc=int(np.sqrt(rho_init.shape[0]))-1 # infer the size of the mode from the size of the initial density matrix
    if nt>1: # if more than one time step
        t_step=t_max/(nt-1) # the size of a single time step
    else: # otherwise just step the whole time at once
        t_step=t_max
    gen_mat=dme.Zeno_blockade_displacement_supOp(gamma_loss,alpha_inj,n_trunc=n_trunc,is_coherent=False,return_gen=True) # construct necessary generator matrix
    if nt>1: # if running for more than one time
        time_step_supOp=linalg_d.expm(t_step*gen_mat) # perform matrix exponentiation to build the superoperator which steps the time
        rho_t=copy.copy(rho_init) # equal to initial density matrix
        func_return_list=[] # empty list to return intermediate values
        for i_t in range(nt): # loop through time steps
            func_return_list=func_return_list+[calculate_func(rho_t)] # calculate values to return
            rho_t=time_step_supOp@rho_t # apply time-stepping superoperator
        return func_return_list
    else:
        rho_t=linalg.expm_multiply(t_step*gen_mat,rho_init) # apply time-stepping superoperator
        return calculate_func(rho_t)
        

def subspace_swap(n_sub,n_before,n_after): # swaps two subspaces to build higher-order operations
    '''
        n_sub is the total size of the subspace to be swapped
        n_before is a list of two variables indicating the total size of the subspace before in the tensor product first being the value before the swap, and the second being the value after the swap
        n_after is a list of two variables indicating the total size of the subspace afterward in the tensor product first being the value before the swap, and the second being the value after the swap
    '''
    
    tot_size=n_before[0]*n_after[0]*n_sub
    input_inds=np.zeros(tot_size)
    output_inds=np.zeros(tot_size)
    n_before_min=min(n_before[0],n_before[1])
    n_after_min=min(n_after[0],n_after[1])
    if n_before[0]==n_before[1]: # special case where no swapping is performed
        return sparse.identity(tot_size,format="csr")
    elif n_before[0]>(n_before[1]*n_sub):
        n_int=int(np.round(n_before[0]/(n_before[1]*n_sub)))
    elif n_before[1]>(n_before[0]*n_sub):
        n_int=int(np.round(n_before[1]/(n_before[0]*n_sub)))
    else:
        n_int=1
    max_vals=np.array([n_before_min,n_sub,n_int,n_sub,n_after_min])-1 # maximum values in each of 5 subspaces, one always before, one always after, one lower swap, one intermediate, one upper swap, and one always above
    subspace_vals=np.zeros(5) # values within each subspace initially
    for i_ind in range(tot_size): # loop through all indices and calculate value for each
        input_inds[i_ind]=subspace_vals[0]+subspace_vals[1]*n_before_min+subspace_vals[2]*n_before_min*n_sub+subspace_vals[3]*n_before_min*n_sub*n_int+subspace_vals[4]*n_before_min*(n_sub**2)*n_int # multiply to calculate new index
        output_inds[i_ind]=subspace_vals[0]+subspace_vals[3]*n_before_min+subspace_vals[2]*n_before_min*n_sub+subspace_vals[1]*n_before_min*n_sub*n_int+subspace_vals[4]*n_before_min*(n_sub**2)*n_int # multiply to calculate new index
        for i_incr in range(5): # increment subspace values
            if subspace_vals[i_incr]==max_vals[i_incr]:
                subspace_vals[i_incr]=0 # reset to zero if it is at the maximum allowed
            else:
                subspace_vals[i_incr]=subspace_vals[i_incr]+1
                break # break out of the inner for loop since it has been incremented
    swap_matrix=sparse.csr_array((np.ones(tot_size),(output_inds,input_inds)),shape=(tot_size,tot_size)) # build sparse matrix
    return swap_matrix


def subspace_move(n_sub,n_before,n_after): # moves subspaces to build higher-order operations
    '''
        n_sub is the total size of the subspace to be moved
        n_before is a list of two variables indicating the total size of the subspace before in the tensor product first being the value before the swap, and the second being the value after the swap
        n_after is a list of two variables indicating the total size of the subspace afterward in the tensor product first being the value before the swap, and the second being the value after the swap
    '''
    tot_size=n_before[0]*n_after[0]*n_sub
    input_inds=np.array(range(tot_size))
    output_inds=np.zeros(tot_size)
    before_div=n_before[1]/n_before[0]
    if before_div>1: # if shifting the subspace to more significant bits
        min_before=n_before[0]
        min_after=n_after[1]
        swap_chunk=before_div
        max_vals=np.array([min_before,n_sub,swap_chunk,min_after])-1 # maximum values each subspace can take
        swap_mult=np.array([1,swap_chunk*min_before,min_before,n_sub*min_before*swap_chunk]) # subspace is swapped to a more significant position
    else:
        min_before=n_before[1]
        min_after=n_after[0]
        swap_chunk=int(1/before_div)
        max_vals=np.array([min_before,swap_chunk,n_sub,min_after])-1 # maximum values each subspace can take
        swap_mult=np.array([1,n_sub*min_before,min_before,n_sub*min_before*swap_chunk]) # subspace is swapped to a less significant position
    subspace_vals=np.array([0,0,0,0]) # values within 4 subspaces, one which is always before, middle subspaces consists of the swapped subspace and the subspace it is swapped with, and one which is always after
    for i_ind in range(tot_size): # loop through all indices
        output_inds[i_ind]=subspace_vals@swap_mult # multiply to calculate new index
        for i_incr in range(4): # increment subspace values
            if subspace_vals[i_incr]==max_vals[i_incr]:
                subspace_vals[i_incr]=0 # reset to zero if it is at the maximum allowed
            else:
                subspace_vals[i_incr]=subspace_vals[i_incr]+1
                break # break out of the inner for loop since it has been incremented
    swap_matrix=sparse.csr_array((np.ones(tot_size),(output_inds,input_inds)),shape=(tot_size,tot_size)) # build sparse matrix
    return swap_matrix
    
def append_identities_Op(Op,add_size): # tensors an additional identity matrix to an operator, defined as a separate function for symmetry with superoperator definitions
    '''
        Op is an operator expressed as an array, its linear size must be a square number, also assumes as square superoperator
        add_size is the size of the subspace to be tensored into each subspace, so the overall matrix size gets multipled by subspace_size**2
    '''
    subspace_size=Op.shape[1]# size of the subspace
    Op_add=QOtrunc.tensor_identities(Op,1,add_size) # append identities
    return Op_add

    
def append_identities_supOp(supOp,add_size): # tensors an additional identity matrix to both the left and right subspace of a superoperator
    '''
        supOp is a superoperator expressed as an array, its linear size must be a square number, also assumes as square superoperator
        add_size is the size of the subspace to be tensored into each subspace, so the overall matrix size gets multipled by subspace_size**2
    '''
    subspace_size=int(np.round(np.sqrt(supOp.shape[1]))) # size of the left and right subspace
    swap_sup=subspace_move(subspace_size,[1,subspace_size],[subspace_size,1]) # matrix to swap the left and right subspace
    supOp_add=swap_sup.T@supOp@swap_sup # swap subspaces
    supOp_add=QOtrunc.tensor_identities(supOp_add,1,add_size) # append identities
    swap_sup_rev=subspace_move(subspace_size,[subspace_size*add_size,1],[1,subspace_size*add_size]) # matrix to swap the left and right subspace back
    supOp_add=swap_sup_rev.T@supOp_add@swap_sup_rev # return the subspace order for enlarged matrix
    supOp_add=QOtrunc.tensor_identities(supOp_add,1,add_size) # append identities
    return supOp_add
    
    
    
def unform_subspace_size_swap(n_sub,sub_count,start_index,end_index): # wrapper to simplify swapping in the usual case of uniform size subspace
    '''
        n_sub is the size of each subspace
        sub_count is the total number of subspaces
        start_index is the index of the targeted subspace
        end_index is the index of the position it is moved to
    '''
    n_before=[n_sub**start_index,n_sub**end_index] # the total size of the subspace before both initially and finally
    n_after=[n_sub**(sub_count-start_index-1),n_sub**(sub_count-end_index-1)] # the total size of the subspace after both initially and finally
    swap_matrix=subspace_swap(n_sub,n_before,n_after)
    return swap_matrix
    
def vaccuum_initialise_rho(n_sub,sub_count): # builds an initial vacuum state for the whole system
    '''
        n_sub is the size of the subspaces
        sub_count is the number of subspaces tensored together
        returns a density matrix represented as a vector
    '''
    total_size=(n_sub**2)**sub_count # total matrix size
    vac_rho_vec=np.zeros(total_size)
    vac_rho_vec[0]=1
    return vac_rho_vec
    
def two_subspace_Op_interaction_from_mat(int_mat,n_sub,sub_count,indices): # builds an interaction between two arbitrary subspaces based on an *operator* matrix defining the interaction,
    '''
        int_mat is an n_sub**2 matrix defining the interaction
        n_sub is the size of each subspace
        sub_count is the total number of subspaces (must be at least 2 to have an interaction
        indices is the two indices where the interaction should be placed, starting with the first subspace
    '''
    int_total=QOtrunc.tensor_identities(int_mat,1,n_sub**(sub_count-2)) # expand to cover whole space
    swap_0=unform_subspace_size_swap(n_sub,sub_count,0,indices[0]) # swap matrix to place first index in correct position
    if not indices[0]==1: # special case if first action swapped first two subspaces
        swap_1=unform_subspace_size_swap(n_sub,sub_count,1,indices[1]) # swap matrix to place second index in correct position
    else:
        swap_1=unform_subspace_size_swap(n_sub,sub_count,0,indices[1]) # use newly swapped position
    int_total=swap_1@swap_0@int_total@swap_0@swap_1 # sandwich between swap operations
    return int_total
    
def two_subspace_supOp_interaction_from_mat(supOp,n_sub,sub_count,indices): # builds an interaction between two arbitrary subspaces based on a *super operator* matrix defining the interaction,
    '''
        supOp is an n_sub**4 matrix defining the interaction
        n_sub is the size of each subspace
        sub_count is the total number of subspaces (must be at least 2 to have an interaction
        indices is the two indices where the interaction should be placed, starting with the first subspace
    '''
    if sub_count==2:
        n_app=1 # special case where subspace is not expanded
    else:
        n_app=n_sub*(sub_count-2)
    int_total=append_identities_supOp(supOp,n_app) # expand to cover whole space
    swap_0=unform_subspace_size_swap(n_sub,sub_count,0,indices[0]) # swap matrix to place first index in correct position
    swap_0_lr=dme.build_supOP(swap_0,swap_0.T) # acting on both left and right subspace
    swap_0_lr_T=dme.build_supOP(swap_0.T,swap_0) # transpose acting on both left and right subspace
    if not indices[0]==1: # special case if first action swapped first two subspaces
        swap_1=unform_subspace_size_swap(n_sub,sub_count,1,indices[1]) # swap matrix to place second index in correct position
    else:
        swap_1=unform_subspace_size_swap(n_sub,sub_count,0,indices[1]) # use newly swapped position
    swap_1_lr=dme.build_supOP(swap_1,swap_1.T) # acting on both left and right subspace
    swap_1_lr_T=dme.build_supOP(swap_1.T,swap_1) # transpose acting on both left and right subspace
    int_total=swap_1_lr_T@swap_0_lr_T@int_total@swap_0_lr@swap_1_lr # sandwich between swap operations
    return int_total
    
def single_subspace_Op_term_from_mat(Op,n_sub,sub_count,index): # construct a superoperator term which acts only on a single subspace
    '''
        Op is an n_sub**2 matrix defining the interaction
        n_sub is the size of each subspace
        sub_count is the total number of subspaces (must be at least 2)
        index is the index at which the single-subspace term will be placed
        returns an operator which applies the operation at the location given by index
    '''
    Op_total=append_identities_Op(Op,n_sub) # append an identity to the operator
    if index==1:
        indices=[index,0] # dummy index in first term
    else:
        indices=[index,1] # dummy index in second term
    term_total=two_subspace_Op_interaction_from_mat(Op_total,n_sub,sub_count,indices) # builtvtotal single subspace term
    return term_total

'''
def single_subspace_supOp_term_from_mat(supOp,n_sub,sub_count,index): # construct a superoperator term which acts only on a single subspace
    '
        supOp is an n_sub**4 matrix defining the interaction
        n_sub is the size of each subspace
        sub_count is the total number of subspaces (must be at least 2)
        index is the index at which the single-subspace term will be placed
        returns a superoperator which applies the operation at the location given by index
    '
    supOp_total=append_identities_supOp(supOp,n_sub) # append an identity to the superoperator
    if index==1:
        indices=[index,0] # dummy index in first term
    else:
        indices=[index,1] # dummy index in second term
    term_total=two_subspace_supOp_interaction_from_mat(supOp_total,n_sub,sub_count,indices) # builtvtotal single subspace term
    return term_total
'''
    
def single_subspace_supOp_term_from_mat(supOp,n_sub,sub_count,index): # construct a superoperator term which acts only on a single subspace
    '''
        supOp is an n_sub**4 matrix defining the interaction
        n_sub is the size of each subspace
        sub_count is the total number of subspaces (must be at least 2)
        index is the index at which the single-subspace term will be placed
        returns a superoperator which applies the operation at the location given by index
    '''
    supOp_total=append_identities_supOp(supOp,n_sub) # append an identity to the superoperator
    if index==1:
        indices=[index,0] # dummy index in first term
    else:
        indices=[index,1] # dummy index in second term
    term_total=two_subspace_supOp_interaction_from_mat(supOp_total,n_sub,sub_count,indices) # builtvtotal single subspace term
    return term_total

def single_mode_supOp_layer(supOp_list,factorised=False): # takes a list of superoperators and makes a layer of superoperators
    '''
        supOp_list is a list of the superoperators applied on each mode, the size of the subspace and number of subspaces are implied from this list
        factorised is a boolean switch, if set to True it returns a list of each superoperator rather than taking product to preserve sparseness
    '''
    n_sub=int(np.round(np.sqrt(supOp_list[0].shape[0]))) # size of the subspace which is acted on by each superoperator (assumend to all be the same)
    sub_count=len(supOp_list) # count of subspaces
    for index_add in range(len(supOp_list)): # loop through to build total superoperator
        if index_add==0: # create superoperator initially
            if factorised:
                supOp_total=[single_subspace_supOp_term_from_mat(supOp_list[index_add],n_sub,sub_count,index_add)] # first element in list
            else:
                supOp_total=single_subspace_supOp_term_from_mat(supOp_list[index_add],n_sub,sub_count,index_add) # first superoperator in product
        else:
            if factorised:
                supOp_total=supOp_total+[single_subspace_supOp_term_from_mat(supOp_list[index_add],n_sub,sub_count,index_add)] # add to list
            else:
                supOp_total=single_subspace_supOp_term_from_mat(supOp_list[index_add],n_sub,sub_count,index_add)@supOp_total # take product to build total superoperator
    return supOp_total
    
def make_forbid_mat_MIS(MIS_adj): # makes a 2**n by 2**n diagonal matrix with 0 and 1 on the diagonal where 1 is an allowed state and 0 is a forbidden one
    '''
        MIS_adj is the adjacency matrix for a maximum-independent set problem
    '''
    n_qubit=MIS_adj.shape[0] # number of binary variables
    total_forbid=np.zeros(2**n_qubit)
    truth_mat=sparse.csr_array([[0,0],[0,1]])
    for i_MIS in range(n_qubit):
        forbid_arr_i=QOtrunc.tensor_identities(truth_mat,2**i_MIS,2**(n_qubit-i_MIS-1)).diagonal()
        for j_MIS in range(i_MIS+1,n_qubit):
            if not MIS_adj[i_MIS,j_MIS]==0:
                forbid_arr_j=QOtrunc.tensor_identities(truth_mat,2**j_MIS,2**(n_qubit-j_MIS-1)).diagonal()
                total_forbid=total_forbid+forbid_arr_i*forbid_arr_j # note that this should be elmentwise multiplication
    keep_elements=np.ones(2**n_qubit)-(total_forbid>0) # array of elements which are not forbidden
    forbid_mat=sparse.csr_array(sparse.dia_array((keep_elements,0),shape=(2**n_qubit,2**n_qubit))) # make sparse array which removes forbidden entries
    return forbid_mat

def single_mode_Op_layer(Op_list,factorised=False,forbid_mat=None): # takes a list of superoperators and makes a layer of operators
    '''
        Op_list is a list of the operators applied on each mode, the size of the subspace and number of subspaces are implied from this list
        factorised is a boolean switch, if set to True it returns a list of each operator rather than taking product to preserve sparseness
        forbid_mat is a matrix used to implement idealised constraints when building a generator layer, set to None and not used by default
    '''
    n_sub=Op_list[0].shape[0] # size of the subspace which is acted on by each operator (assumend to all be the same)
    sub_count=len(Op_list) # count of subspaces
    for index_add in range(len(Op_list)): # loop through to build total superoperator
        if index_add==0: # create operator initially
            if factorised:
                Op_total=[single_subspace_Op_term_from_mat(Op_list[index_add],n_sub,sub_count,index_add)] # first element in list
            else:
                Op_total=single_subspace_Op_term_from_mat(Op_list[index_add],n_sub,sub_count,index_add) # first superoperator in product
        else:
            if factorised:
                Op_total=Op_total+[single_subspace_Op_term_from_mat(Op_list[index_add],n_sub,sub_count,index_add)] # add to list
            else:
                Op_total=single_subspace_Op_term_from_mat(Op_list[index_add],n_sub,sub_count,index_add)@Op_total # take product to build total operator
    if not type(forbid_mat)==type(None): # if a matrix is supplied to forbid some states
        if factorised:
            Op_total_orig=copy.copy(Op_total)
            Op_total=[]
            for Op in Op_total_orig: # loop through previously created generators and forbid certain states
                Op_total=Op_total+[forbid_mat@Op@forbid_mat] # diagonal so self transpose
        else:
            warnings.warn('a forbid_mat has been supplied with the non-factorized option, which implies that operators rather than generators are supplied, this will lead to non-unitary behavior which deviates from the ideal')
            Op_total=forbid_mat@Op_total@forbid_mat
    return Op_total
    
def construct_interaction_Op_network_from_mat(int_mat,n_sub,sub_count,indices_list): # builds a network of interactions based on a two-subspace interaction matrix
    '''
        int_mat is an n_sub**2 matrix defining the interaction superoperator
        n_sub is the size of each subspace
        sub_count is the total number of subspaces (must be at least 2 to have an interaction
        indices_list is a list of index pairs)
    '''
    if len(indices_list)==0:
        return sparse.identity(n_sub**(sub_count),dtype='complex',format='csr')
    for i_network in range(len(indices_list)):
        int_add=two_subspace_Op_interaction_from_mat(int_mat,n_sub,sub_count,indices_list[i_network])
        if i_network==0:
            int_total=copy.copy(int_add)
        else:
            int_total=int_add@int_total
    return int_total
    
def construct_interaction_supOp_network_from_mat(int_mat,n_sub,sub_count,indices_list): # builds a network of interactions based on a two-subspace interaction matrix
    '''
        int_mat is an n_sub**4 matrix defining the interaction superoperator
        n_sub is the size of each subspace
        sub_count is the total number of subspaces (must be at least 2 to have an interaction
        indices_list is a list of index pairs)
    '''
    if len(indices_list)==0:
        return sparse.identity(n_sub**(2*sub_count),dtype='complex',format='csr')
    for i_network in range(len(indices_list)):
        int_add=two_subspace_supOp_interaction_from_mat(int_mat,n_sub,sub_count,indices_list[i_network])
        if i_network==0:
            int_total=copy.copy(int_add)
        else:
            int_total=int_add@int_total
    return int_total
    
def construct_interaction_supOp_network_from_mat(int_mat,n_sub,sub_count,indices_list): # builds a network of interactions based on a two-subspace interaction matrix
    '''
        int_mat is an n_sub**4 matrix defining the interaction superoperator
        n_sub is the size of each subspace
        sub_count is the total number of subspaces (must be at least 2 to have an interaction
        indices_list is a list of index pairs)
    '''
    if len(indices_list)==0:
        return sparse.identity(n_sub**(2*sub_count),dtype='complex',format='csr')
    for i_network in range(len(indices_list)):
        int_add=two_subspace_supOp_interaction_from_mat(int_mat,n_sub,sub_count,indices_list[i_network])
        if i_network==0:
            int_total=copy.copy(int_add)
        else:
            int_total=int_add@int_total
    return int_total
    
def construct_drive_interaction_Op_network_from_mat(int_mat,drive_mats,n_sub,sub_count,indices_list,driving_sequence='before'): # builds a network of interactions based on a two-subspace interaction matrix
    '''
        int_mat is an n_sub**2 matrix defining the interaction operator
        n_sub is the size of each subspace
        drive_mats is a list of all driving operators, tensored with identities so they act over the whole space
        sub_count is the total number of subspaces (must be at least 2 to have an interaction
        indices_list is a list of index pairs)
        driving_sequence determines when the driving is perfomed, options are 'before', 'after', or 'both'
    '''
    if driving_sequence=='ideal':
        int_total=sparse.identity(n_sub**(2*sub_count),dtype='complex',format='csr')
        for i_drive in len(drive_mats):
            int_total=int_total@drive_mats[i_drive] # multupy in driving mat
        return int_total
    if len(indices_list)==0:
        return sparse.identity(n_sub**(2*sub_count),dtype='complex',format='csr')
    for i_network in range(len(indices_list)):
        drive_add=drive_mats[indices_list[i_network][1]]@drive_mats[indices_list[i_network][0]]
        int_add=two_subspace_Op_interaction_from_mat(int_mat,n_sub,sub_count,indices_list[i_network])
        if driving_sequence=='before':
            tot_add=int_add@drive_add
        elif driving_sequence=='after':
            tot_add=drive_add@int_add
        elif driving_squence=='both':
            tot_add=drive_add@int_add@drive_add
        else:
            raise RuntimeError("unrecognised driving sequence: '"+driving_sequence+"' allowed options are 'before', 'after' or 'both'")
        if i_network==0:
            int_total=copy.copy(tot_add)
        else:
            int_total=tot_add@int_total
    return int_total

def construct_drive_interaction_supOp_network_from_mat(int_mat,drive_mats,n_sub,sub_count,indices_list,driving_sequence='before'): # builds a network of interactions based on a two-subspace interaction matrix
    '''
        int_mat is an n_sub**4 matrix defining the interaction superoperator
        n_sub is the size of each subspace
        drive_mats is a list of all driving operators, tensored with identities so they act over the whole space
        sub_count is the total number of subspaces (must be at least 2 to have an interaction
        indices_list is a list of index pairs)
        driving_sequence determines when the driving is perfomed, options are 'before', 'after', or 'both'
    '''
    if len(indices_list)==0:
        return sparse.identity(n_sub**(2*sub_count),dtype='complex',format='csr')
    for i_network in range(len(indices_list)):
        drive_add=drive_mats[indices_list[i_network][1]]@drive_mats[indices_list[i_network][0]]
        int_add=two_subspace_supOp_interaction_from_mat(int_mat,n_sub,sub_count,indices_list[i_network])
        if driving_sequence=='before':
            tot_add=int_add@drive_add
        elif driving_sequence=='after':
            tot_add=drive_add@int_add
        elif driving_squence=='both':
            tot_add=drive_add@int_add@drive_add
        else:
            raise RuntimeError("unrecognised driving sequence: '"+driving_sequence+"' allowed options are 'before', 'after' or 'both'")
        if i_network==0:
            int_total=copy.copy(tot_add)
        else:
            int_total=tot_add@int_total
    return int_total
    
def construct_gen_dict_drive_interaction_supOp(int_mat,drive_gens,n_sub,sub_count,indices_list): # function to pre-compute all necessary tensor products so that none are performed in the inner loop
    '''
        int_mat is an n_sub**4 matrix defining the interaction superoperator
        n_sub is the size of each subspace
        drive_gens is a list of all driving generators, not yet tensored with identities assumed to already be normalised by degree
        sub_count is the total number of subspaces (must be at least 2 to have an interaction
        indices_list is a list of index pairs)
        driving_sequence determines when the driving is perfomed, options are 'before', 'after', or 'both'
    '''
    gen_dict={} # empty dictionary
    if not type(drive_gens)==type({}): # if supplied as a list rather than a dictionary
        gen_dict['drive_gens']=single_mode_supOp_layer(drive_gens,factorised=True) # build a layer of superoperators containing all the driving generators
    else:
        gen_dict['drive_gens']=drive_gens # otherwise just pass the dictionary along
    gen_dict['indices_list']=indices_list # add list of indices to dictionary
    gen_dict['int_list']=[] # empty list to populate with interaction superoperators
    for i_network in range(len(indices_list)): # loop through interactions
        gen_dict['int_list']=gen_dict['int_list']+[two_subspace_supOp_interaction_from_mat(int_mat,n_sub,sub_count,indices_list[i_network])] # add to list of interaction superoperators
    return gen_dict
    
def construct_normalised_drive_gen_list_Op(MIS_adj,drive_H,driving_sequence='before'): # generates a list of normalised generators for driving operators
    '''
        MIS_adj is the adjacency matrix for a maximum independent set problem
        drive_gens is a list of all driving generators, not yet tensored with identities assumed to already be normalised by degree
        driving_sequence determines when the driving is perfomed, options are 'before', 'after', 'ideal' or 'both'
    '''
    adj_sym=np.triu(MIS_adj)+np.triu(MIS_adj).T # make the adjacency matrix symmetric
    degree_arr=sum(adj_sym) # the degreee of the adjacency matrix
    drive_gens=[] # empty list to populate with generators
    for i_degree in range(len(degree_arr)):
        if driving_sequence=='before' or driving_sequence=='after': # divide by degree
            drive_gens=drive_gens+[-1j*drive_H/degree_arr[i_degree]] # multiply by -1j to produce generator of dynamics from Hamiltonian
        elif driving_sequence=='both': # divide by twice the degree
            drive_gens=drive_gens+[-1j*drive_H/(2*degree_arr[i_degree])] # multiply by -1j to produce generator of dynamics from Hamiltonian
        elif driving_sequence=='monolithic' or driving_sequence=='ideal': # don't normalise by degree
            drive_gens=drive_gens+[-1j*drive_H] # multiply by -1j to produce generator of dynamics from Hamiltonian
        else:
            raise RuntimeError("unrecognised driving sequence: '"+driving_sequence+"' allowed options are 'before', 'after', 'ideal', or 'both'")
    return drive_gens
    
def construct_gen_dict_drive_interaction_Op(int_mat,drive_gens,n_sub,sub_count,indices_list,forbid_mat=None): # function to pre-compute all necessary tensor products so that none are performed in the inner loop
    '''
        int_mat is an n_sub**2 matrix defining the interaction operator
        n_sub is the size of each subspace
        drive_gens is a list of all driving generators, not yet tensored with identities assumed to already be normalised by degree
        sub_count is the total number of subspaces (must be at least 2 to have an interaction
        indices_list is a list of index pairs)
        forbid_mat is a matrix to forbid certain configurations, used when an idealised version of driving is employed
    '''
    gen_dict={} # empty dictionary
    gen_dict['drive_gens']=single_mode_Op_layer(drive_gens,factorised=True,forbid_mat=forbid_mat) # build a layer of superoperators containing all the driving generators
    gen_dict['indices_list']=indices_list # add list of indices to dictionary
    gen_dict['int_list']=[] # empty list to populate with interaction superoperators
    for i_network in range(len(indices_list)): # loop through interactions
        gen_dict['int_list']=gen_dict['int_list']+[two_subspace_Op_interaction_from_mat(int_mat,n_sub,sub_count,indices_list[i_network])] # add to list of interaction superoperators
    return gen_dict
    
def construct_drive_interaction_supOp_network_from_gen_dict(gen_dict,tot_rotation,driving_sequence='before'): # builds a network of interactions based on a two-subspace interaction matrix
    '''
        gen_dict is a precomputed dictionary constructed to avoid tensor products and swapping in the inner loop, can also contain precomputed elements to avoid matrix exponentiation in inner loop
        tot_rotation is the total rotation to be performed
        driving_sequence determines when the driving is perfomed, options are 'before', 'after', or 'both'
    '''
    indices_list=gen_dict['indices_list']
    if type(gen_dict['drive_gens'])==type({}): # if provided as a dictionary, use precomputed elements to avoid matrix exponentiation in inner loop
        drive_supOps=construct_driving_from_dict(gen_dict['drive_gens'],tot_rotation) # construct list of superoperators
    if len(indices_list)==0:
        return sparse.identity(n_sub**(2*sub_count),dtype='complex',format='csr')
    for i_network in range(len(indices_list)):
        if not type(gen_dict['drive_gens'])==type({}): # if not provided as a dictionary
            drive_add=linalg.expm(gen_dict['drive_gens'][indices_list[i_network][0]]*tot_rotation)
            drive_add=linalg.expm(gen_dict['drive_gens'][indices_list[i_network][1]]*tot_rotation)@drive_add
        else: # avoid matrix exponentiation in inner lo
            drive_add=drive_supOps[indices_list[i_network][0]]
            drive_add=drive_supOps[indices_list[i_network][1]]@drive_add
        int_add=gen_dict['int_list'][i_network]
        if driving_sequence=='before':
            tot_add=int_add@drive_add
        elif driving_sequence=='after':
            tot_add=drive_add@int_add
        elif driving_squence=='both':
            tot_add=drive_add@int_add@drive_add
        else:
            raise RuntimeError("unrecognised driving sequence: '"+driving_sequence+"' allowed options are 'before', 'after' or 'both'")
        if i_network==0:
            int_total=copy.copy(tot_add)
        else:
            int_total=tot_add@int_total
    return int_total

def construct_drive_interaction_Op_network_from_gen_dict(gen_dict,tot_rotation,driving_sequence='before'): # builds a network of interactions based on a two-subspace interaction matrix
    '''
        gen_dict is a precomputed dictionary constructed to avoid tensor products and swapping in the inner loop, can also contain precomputed elements to avoid matrix exponentiation in inner loop
        tot_rotation is the total rotation to be performed
        driving_sequence determines when the driving is perfomed, options are 'before', 'after', 'ideal' or 'both'
    '''
    indices_list=gen_dict['indices_list']
    sub_count=len(gen_dict['drive_gens'])
    n_sub=2
    if driving_sequence=='ideal':
        int_total=np.eye(n_sub**(sub_count))
        for i_drive in range(len(gen_dict['drive_gens'])):
            int_total=linalg_d.expm(gen_dict['drive_gens'][i_drive]*tot_rotation)@int_total
        return int_total
    if len(indices_list)==0:
        return np.eye(n_sub**(sub_count))
    for i_network in range(len(indices_list)):
        drive_add=linalg_d.expm(gen_dict['drive_gens'][indices_list[i_network][0]]*tot_rotation)
        drive_add=linalg_d.expm(gen_dict['drive_gens'][indices_list[i_network][1]]*tot_rotation)@drive_add
        int_add=gen_dict['int_list'][i_network]
        if driving_sequence=='before':
            tot_add=int_add@drive_add
        elif driving_sequence=='after':
            tot_add=drive_add@int_add
        elif driving_sequence=='both':
            tot_add=drive_add@int_add@drive_add
        else:
            raise RuntimeError("unrecognised driving sequence: '"+driving_sequence+"' allowed options are 'before', 'after', 'ideal', or 'both'")
        if i_network==0:
            int_total=copy.copy(tot_add)
        else:
            int_total=tot_add@int_total
    return int_total
    
def index_to_subspace_values(index_val,subspace_sizes): # converts an index to values taken within different subspaces
    '''
        index_val is the value of the index
        subspace_sizes is the size of all subspaces in the overall Hilbert space
    '''
    subspace_values=np.zeros(len(subspace_sizes),dtype=int) # array of all zeros to add the values in each subspace
    total_size=np.prod(np.array(subspace_sizes)) # total size of all subspaces (to check that index is within bounds)
    working_val=copy.copy(index_val)
    if not index_val<total_size: # if not within the allowed range based on the sizes
        warnings.warn("index exceeds maximum size: rounding off more significant subspaces, if the index was not intended to be larger than the total sizes of subspaces this indicates a problem") # warning in case the index was not meant to exceed the values
        working_val=working_val%total_size # take mod
    for i_sub in range(len(subspace_sizes)):
        after_size=np.prod(np.array(subspace_sizes)[(i_sub+1):]) # product of all subsequent values
        subspace_values[len(subspace_sizes)-i_sub-1]=int(np.floor(working_val/after_size)) # take floor to calcualte value
        working_val=working_val%after_size # take mod to remove most signficant subspace
    return subspace_values

def optimal_phase_schedule(n_step,max_phase=np.pi/2): # generates an optimal (for a single qubit) schedule of phase applications based on the single avoided crossing model
    '''
        n_step is the numeber of discrete timesteps to be taken
        max_phase is the maximum maginitude of phase to be applied, defaults to np.pi which corresponds to a full phase inversion at each step
        returns a 1D array of length n_step which is the optimal schedule
    '''
    normalisation=abs(max_phase/np.tan(np.pi/(n_step+1)-np.pi/2))
    sched=normalisation*np.tan(np.pi*np.linspace(1/(n_step+1),1-1/(n_step+1),n_step)-np.pi/2)
    return sched
    
def optimal_phase_and_disp_schedule_norm(n_step,rotation_per_stage=1): # generates an optimal (for a single qubit) schedule of phase applications based on the single avoided crossing model
    '''
        n_step is the numeber of discrete timesteps to be taken
        rotation_per_stage is the sum of the phase and displacement rotation performed each time
    '''
    ratio_sched=np.tan(np.pi*np.linspace(1/(n_step+1),1-1/(n_step+1),n_step)-np.pi/2)
    phase_sched=rotation_per_stage*ratio_sched/(1+abs(ratio_sched))
    disp_sched=rotation_per_stage/(1+abs(ratio_sched))
    return (phase_sched,disp_sched)
    
def phase_supOp_0_1_pos_and_neg_inds(num_qubits): # function to determine indices for positive and negative rotation from applying a phase operator to each qubit, these are used to construct a phase layer without having to perform tensor products or swapping
    '''
        num_qubits is the number of qubits
        returns a list of arrays containing diagonal indices where positive or negative rotations are applied
    '''
    phase_supOp=dme.apply_phase_shift_zero_one_supOp(np.pi/2) # define superoperator
    pos_inds_list=[]
    neg_inds_list=[]
    for phase_ind in range(num_qubits):
        phase_supOp_list=[] # empty list
        for i_prod in range(num_qubits): # calculate phase superoperators
            if i_prod==phase_ind:
                phase_supOp_list=phase_supOp_list+[copy.copy(phase_supOp)] # add to list
            else:
                phase_supOp_list=phase_supOp_list+[sparse.eye_array(4)] # add to list
        phase_layer=single_mode_supOp_layer(phase_supOp_list) # build phase layer for this stage
        pos_inds_list=pos_inds_list+[np.where(phase_layer.diagonal()==1j)[0]] # indices with positive phase rotation
        neg_inds_list=neg_inds_list+[np.where(phase_layer.diagonal()==-1j)[0]] # indices with negative phase rotation
    return (pos_inds_list,neg_inds_list)
    
def phase_Op_0_1_pos_and_neg_inds(num_qubits): # function to determine indices for positive and negative rotation from applying a phase operator to each qubit, these are used to construct a phase layer without having to perform tensor products or swapping
    '''
        num_qubits is the number of qubits
        returns a list of arrays containing diagonal indices where positive or negative rotations are applied
    '''
    phase_Op=np.array([[1j,0],[0,-1j]]) # define operator
    pos_inds_list=[]
    neg_inds_list=[]
    for phase_ind in range(num_qubits):
        phase_Op_list=[] # empty list
        for i_prod in range(num_qubits): # calculate phase superoperators
            if i_prod==phase_ind:
                phase_Op_list=phase_Op_list+[copy.copy(phase_Op)] # add to list
            else:
                phase_Op_list=phase_Op_list+[sparse.eye_array(2)] # add to list
        phase_layer=single_mode_Op_layer(phase_Op_list) # build phase layer for this stage
        pos_inds_list=pos_inds_list+[np.where(phase_layer.diagonal()==1j)[0]] # indices with positive phase rotation
        neg_inds_list=neg_inds_list+[np.where(phase_layer.diagonal()==-1j)[0]] # indices with negative phase rotation
    return (pos_inds_list,neg_inds_list)

def build_phase_layer_supOp_0_1_from_indices(phi_vals,num_qubits,pos_inds_list,neg_inds_list): # uses pre-computed indices to build a phase superoperator rather than tensor products to avoid numerical bottlenecks
    '''
        phi_vals an array of the values of the phases to be applied, or a single value if all the same
        num_qubits is the number of qubits
        pos_ind_list is the list of arrays of indices for postive rotations for each qubit
        neg_ind_list is the list of arrays of indices for negative rotations for each qubit
    '''
    diag_arr=np.ones(4**num_qubits,dtype=complex) # initial diagonal before any phases are applied
    if not type(phi_vals)==np.ndarray: # if we are supplied a single value
        phase_pos=np.exp(1j*phi_vals) # compute once for later use
        phase_neg=np.exp(-1j*phi_vals)
    for i_qubit in range(num_qubits): # loop through the phase applied to each qubit
        if type(phi_vals)==np.ndarray: # if we are supplied an array
            diag_arr[pos_inds_list[i_qubit]]=np.exp(1j*phi_vals[i_qubit])*diag_arr[pos_inds_list[i_qubit]] # apply positive phase in necessary locations
            diag_arr[neg_inds_list[i_qubit]]=np.exp(-1j*phi_vals[i_qubit])*diag_arr[neg_inds_list[i_qubit]] # apply negative phase in necessary locations
        else:
            diag_arr[pos_inds_list[i_qubit]]=phase_pos*diag_arr[pos_inds_list[i_qubit]] # apply positive phase in necessary locations
            diag_arr[neg_inds_list[i_qubit]]=phase_neg*diag_arr[neg_inds_list[i_qubit]] # apply negative phase in necessary locations
    phase_layer=sparse.spdiags(diag_arr,0,4**num_qubits,4**num_qubits) # use the diagonal to construct a sparse array
    return phase_layer
    
def build_phase_layer_Op_0_1_from_indices(phi_vals,num_qubits,pos_inds_list,neg_inds_list): # uses pre-computed indices to build a phase operator rather than tensor products to avoid numerical bottlenecks
    '''
        phi_vals an array of the values of the phases to be applied, or a single value if all the same
        num_qubits is the number of qubits
        pos_ind_list is the list of arrays of indices for postive rotations for each qubit (trivial phase applied, but kept to preserve symmetry with superoperator case)
        neg_ind_list is the list of arrays of indices for negative rotations for each qubit
    '''
    diag_arr=np.ones(2**num_qubits,dtype=complex) # initial diagonal before any phases are applied
    if not type(phi_vals)==np.ndarray: # if we are supplied a single value
        phase_pos=1 # compute once for later use
        phase_neg=np.exp(1j*phi_vals)
    for i_qubit in range(num_qubits): # loop through the phase applied to each qubit
        if type(phi_vals)==np.ndarray: # if we are supplied an array
            diag_arr[pos_inds_list[i_qubit]]=diag_arr[pos_inds_list[i_qubit]] # apply positive phase in necessary locations
            diag_arr[neg_inds_list[i_qubit]]=np.exp(1j*phi_vals[i_qubit])*diag_arr[neg_inds_list[i_qubit]] # apply negative phase in necessary locations
        else:
            diag_arr[pos_inds_list[i_qubit]]=phase_pos*diag_arr[pos_inds_list[i_qubit]] # apply positive phase in necessary locations
            diag_arr[neg_inds_list[i_qubit]]=phase_neg*diag_arr[neg_inds_list[i_qubit]] # apply negative phase in necessary locations
    phase_layer=sparse.spdiags(diag_arr,0,2**num_qubits,2**num_qubits) # use the diagonal to construct a sparse array
    return phase_layer
    
def two_mode_0_1_HOM_nl_apply(gamma=np.pi/(2*np.sqrt(2)),phase_apply=0): # uses Hong Ou Mandel effect to place a pair of photons in one mode and then uses non-linearity to apply phases or loss
    '''
        gamma is the total amount of non-linear evolution applied, with np.pi/(2*np.sqrt(2)) corresponding to complete removal and return
        phase_apply is a variable which determines what phase is applied to the pump mode before it returns, defaults to 0
    '''
    bs5050_Op2=linalg_d.expm(QOtrunc.apply_beamsplitting_Op(np.pi/4,9,[3,1],[1,3],gen_mat=True).todense()) # 50:50 beamsplitting operator acting on modes with up to two photons
    bs5050_supOp2=dme.build_supOP(bs5050_Op2,bs5050_Op2.conj().T) # beamsplitting superoperator acting on modes with up to two photons
    bs5050inv_Op2=linalg_d.expm(QOtrunc.apply_beamsplitting_Op(-np.pi/4,9,[3,1],[1,3],gen_mat=True).todense()) # 50:50 beamsplitting operator acting on modes with up to one photon, phases reversed to cancel exactly
    bs5050inv_supOp2=dme.build_supOP(bs5050inv_Op2,bs5050inv_Op2.conj().T) # phase inverted beamsplitting superoperator acting on modes with up to one photon
    mode_expand_Op1_2=dme.make_subspace_expand_matrix([2,2],[3,3]) # operator which expands from one two two photons
    mode_expand_supOp1_2=dme.build_supOP(mode_expand_Op1_2,mode_expand_Op1_2.T) # superoperator for expanding Hilbert spaces
    nl_supOp_1mode=dme.expand_nl_phase_nl_trace_supOP(gamma=gamma,phase_apply=phase_apply,n_trunc=2) # superoperator acting on a single mode
    nl_supOp_layer=single_mode_supOp_layer([nl_supOp_1mode,nl_supOp_1mode],factorised=False)
    supOp_HOM=mode_expand_supOp1_2.T@bs5050inv_supOp2@nl_supOp_layer@bs5050_supOp2@mode_expand_supOp1_2 # chain Hilbert space expansion, beamsplitting, nonlinear operations in expanded Hilbert spaces on each mode, and beamsplitting as well as Hilbert space contraction to complete operation
    return supOp_HOM
    

    
def make_normlised_driving_supOps(drive_func,MIS_adj,total_disp): # makes a layer of driving superoperators (or operartors) which are normalised for interspersed operation
    '''
        drive_func is a function which is supplied to calculate the driving strength, it takes one argument, which is the amount of displacement
        MIS_adj is the adjacency matrix of a graph on which the independent set is to be found given as an array (only uses the upper triangular portion)
        total_disp is the total amount of dispalcement per layer
    '''
    adj_sym=np.triu(MIS_adj)+np.triu(MIS_adj).T # make the adjacency matrix symmetric
    degree_arr=sum(adj_sym) # the degreee of the adjacency matrix
    driver_list=[] # empty list
    for i_qubit in range(len(degree_arr)): # loop to create superoperators
        driver_list=driver_list+[drive_func(total_disp/degree_arr[i_qubit])] # add new driver to list
    return driver_list # return the list of all drivers
    

def make_supOp_driving_construction_dict(supOp_gen,MIS_adj,mult_before=None,mult_after=None,driving='before'): # makes the superoperators for constructing driving
    '''
        supOp_gen is the generator for the superoperator
        MIS_adj is the adjacency matrix for the graph
        mult_before is a matrix to be multiplied before the superoperator (normally to expand the subspace) defaults to none
        mult_after is a matrix to be multiplied after the superoperator (normally to project down to a smaller subspace) defaults to none
        driving is how the driving will be performed, options are 'before', 'after', 'both', or 'monolithic'
        returns a dictionary with necessary fields to construct the driving matrix
    '''
    supOp_construct_dict={} # start with empty dictionary
    num_qubits=MIS_adj.shape[0] # number of qubits used in computation
    supOp_construct_dict['kron_precomputes']=[] # empty list
    adj_sym=np.triu(MIS_adj)+np.triu(MIS_adj).T # make the adjacency matrix symmetric
    degree_arr=sum(adj_sym) # the degreee of the adjacency matrix
    if driving=='monolithic':
        supOp_construct_dict['exp_mult']=1 # single value when performed monolithically
    else:
        supOp_construct_dict['exp_mult']=[] # empty list
    supOp_construct_dict['expm_precompute']=dme.construct_expm_binary_approx(supOp_gen,max_exp=np.pi,num_powers=30)
    for i_qubit in range(num_qubits): # loop through and create precomputed matrices
        size_before=2**i_qubit # the size of the previous subspace
        size_after=2**(num_qubits-i_qubit-1) # the size of the previous subspace
        supOp_construct_dict['kron_precomputes']=supOp_construct_dict['kron_precomputes']+[dme.compute_tensor_expansion_matrices_supOp(2,size_before,size_after)] # add to the precomputed projectors
        if driving=='before' or driving=='after':
            supOp_construct_dict['exp_mult']=supOp_construct_dict['exp_mult']+[1/degree_arr[i_qubit]] # divide and add to list of generators
        elif driving=='both':
            supOp_construct_dict['exp_mult']=supOp_construct_dict['exp_mult']+[1/(2*degree_arr[i_qubit])] # extra factor of 2 if both before and after
        supOp_construct_dict['mult_before']=mult_before
    if not type(mult_after)==type(None):
        supOp_construct_dict['mult_after']=mult_after
    return supOp_construct_dict
        
    
def construct_driving_from_dict(construct_dict,total_drive): # uses a dictionary to construct custom drivers, with an overall scaling of the rotation performed defined by total_drive
    '''
        construct_dict is a dictionary which contains the necessary components to construct driver terms (note this function will work for both operators and superoperators
        required fields are:
            exp_mults modewise multipliers to give different powers on each mode, set to 1 when doing monolithic driving
            kron_precomputes pre-computed matrices to expand to the whole space list of sparse arrays equal to the number of qubits
        optional fields are:
            mult_before dense array to multiply the exponentiated matrix from the right
            mult_after dense array to multiply the exponentiated matrix from the left
        total_drive is the total strength of the driving
    '''
    drive_list=[] # empty list of drivers
    for iDrive in range(len(construct_dict['kron_precomputes'])): # component on which to perform numerical matrix exponentiation
        if type(construct_dict['exp_mult'])==type([]):
            driver=dme.fast_expm_from_dict(construct_dict['expm_precompute'],construct_dict['exp_mult'][iDrive]*total_drive) # multiply precomputed matrices to speed up the process
        elif iDrive==0: # compute once since they will all be identical
            driver_init=dme.fast_expm_from_dict(construct_dict['expm_precompute'],construct_dict['exp_mult']*total_drive) # master copy to use multiple times
            driver=copy.copy(driver_init) # copy exponentiatied matrix
        else:
            driver=copy.copy(driver_init) # copy exponentiatied matrix
        if 'mult_before' in construct_dict: # if it gets projected
            driver=driver@construct_dict['mult_before']
        if 'mult_after' in construct_dict: # if it gets projected
            driver=construct_dict['mult_after']@driver
        kron_precompute=construct_dict['kron_precomputes'][iDrive] # use precomputed matrices to avoid tensor products in the inner loop
        drive_list=drive_list+[dme.build_tensor_product_from_expand_mat_list(sparse.csr_array(driver),kron_precompute)] # multiply out to cover the whole space after converting to sparse
    return drive_list
    
def optical_anneal_wmis(supOp_drive,phase_schedule,MIS_adj,driving='monolithic',driving_schedule=None,interaction_supOp=dme.two_mode_0_1_HOM(),phase_weights=None,intermediate_calc_functions=None): # applies an anneal to solve a weighted maximum independent set in a density matrix superoperator representation
    '''
        supOp_drive is a single subspace superoperator (or a dictionary which can be used to quickly construct a superoperator) which drives the system, can be supplied as a list to give different driving on different modes
        phase_schedule is an iterable containing all the phases applied at different point
        MIS_adj is the adjacency matrix of a graph on which the independent set is to be found given as an array (only uses the upper triangular portion)
        driving determines how the driving is done, options are 'monolithic' for a single layer of driving, 'before' for interespered with interaction before each, 'after' for interspersed with interactions after each, and 'both' for interspersed with interaction both before and after
        driving_schedule deterimines how the driving changes over time, if set to None we just assume driving is constant with the supplied superoperator, otherwise it is assumed that a generator is supplied
        phase_weights is an array of positive weights on each node, defaults to None in which case an unweighted problem is solved
        intermediate_calc_function is a function which calculates statistics for intermediate points in evolution, defaults to None, in which case nothing is applied
    '''
    if not type(driving_schedule)==type(None) and not type(supOp_drive)==type({}):
        warnings.warn('a driving schedule has been provided but the drver has not been provided as a dictionary for precomputation, this function can handle this combination of inputs but it performs matrix exponentiation in the inner loop and is therefore likely to be slow')
    num_qubits=MIS_adj.shape[0] # the number of qubit subspaces to be used
    (pos_inds_list,neg_inds_list)=phase_supOp_0_1_pos_and_neg_inds(num_qubits) # pre-computed diagonal indices to make inner loop faster
    rho_vec_t=np.zeros(4**num_qubits)
    rho_vec_t[0]=1 # initialise as vacuum
    if not type(supOp_drive)==type({}): # the driver is supplied as a matrix rather than a construction dictionary
        if not type(supOp_drive)==type([]): # if a single superoperator rather than a list of superoperators
            drive_layer_supOps=single_mode_supOp_layer([supOp_drive]*num_qubits,factorised=True) # make a superoperator for the driving layer
        else:
            drive_layer_supOps=single_mode_supOp_layer(supOp_drive,factorised=True) # make a superoperator for driving based on supplied list
    else: # otherwise perform precomputation to allow the drivers to be built quickly within the inner loop
        drive_layer_supOps=make_supOp_driving_construction_dict(supOp_drive['supOp_gen'],MIS_adj,mult_before=supOp_drive['mult_before'],mult_after=supOp_drive['mult_after'],driving=driving) #  precompute some elements to allow the matrix to be constructed faster
    indices_list=[]
    for i_MIS in range(num_qubits): # first adjacency index
        for j_MIS in range(i_MIS+1,num_qubits): # second adjacency index
            if not MIS_adj[i_MIS,j_MIS]==0: # if a non-zero entry in the adjacency matrix
                indices_list=indices_list+[[i_MIS,j_MIS]] # add to list of indices to create network to apply independence condition
    if driving=='monolithic': # driving done separately in monolithic case
        ind_layer=construct_interaction_supOp_network_from_mat(interaction_supOp,2,num_qubits,indices_list)
    else: # otherwise layers are combined
        if type(driving_schedule)==type(None): # for constant driving compute the layer superoperator once at the beginning
            gen_dict=construct_gen_dict_drive_interaction_supOp(interaction_supOp,drive_layer_supOps,2,num_qubits,indices_list)
            ind_layer=construct_drive_interaction_supOp_network_from_gen_dict(gen_dict,1,driving_sequence=driving)
            #ind_layer=construct_drive_interaction_supOp_network_from_mat(interaction_supOp,drive_layer_supOps,2,num_qubits,indices_list,driving_sequence=driving)
        else: # for variable driving perform tensor product and swapping pre-computations
            gen_dict=construct_gen_dict_drive_interaction_supOp(interaction_supOp,drive_layer_supOps,2,num_qubits,indices_list)
    intermediate_data_list=[] # empty list for intermediate data
    for iStage in range(len(phase_schedule)): # each stage of evolution
        if type(phase_weights)==type(None): # if undefined, then create array of equal weights
            phase_layer=build_phase_layer_supOp_0_1_from_indices(phase_schedule[iStage],num_qubits,pos_inds_list,neg_inds_list) # construct phase layer using pre-computed
        else: # otherwise calculate an individual superoperator for each mode
            phase_layer=build_phase_layer_supOp_0_1_from_indices(phase_schedule[iStage]*phase_weights,num_qubits,pos_inds_list,neg_inds_list) # construct phase layer using pre-computed indices to avoid tensor products in the inner loop
        if driving=='monolithic': # driving done separately in monolithic case
            if type(supOp_drive)==type({}): # if provided as a dictionary
                if type(driving_schedule)==type(None) and iStage==0: # compute only once if no schedule
                    drive_apply=construct_driving_from_dict(drive_layer_supOps,supOp_drive['drive_strength'])
                else: # compute every time from precomputed data if a schedule is provided
                    drive_apply=construct_driving_from_dict(drive_layer_supOps,driving_schedule[iStage])
            else: # if not provided as a dictionary just pass to next loop
                drive_apply=drive_layer_supOps
            for drive_supOp in drive_apply: # multiply one-by-one to avoid dense matrices
                if type(driving_schedule)==type(None) or type(supOp_drive)==type({}): # if already computed from dictionary, or no driving schedule provided
                    rho_vec_t=drive_supOp@rho_vec_t # apply driving within one qubit subspace
                else:
                    rho_vec_t=linalg.expm_multiply(drive_supOp*driving_schedule[iStage],rho_vec_t) # apply driving within one qubit subspace
        else:
            if (not type(driving_schedule)==type(None)): # final construction within inner loop if driving varied over time
                ind_layer=construct_drive_interaction_supOp_network_from_gen_dict(gen_dict,driving_schedule[iStage],driving_sequence=driving)
        rho_vec_t=phase_layer@rho_vec_t # apply phase layer (diagonal so we can apply all at once)
        rho_vec_t=ind_layer@rho_vec_t # apply layer to enforce independence (sparse so we can apply all at once)
        if not type(intermediate_calc_functions)==type(None):  # if intermediate values are to be computed
            intermediate_data_list=intermediate_data_list+[intermediate_calc_functions(rho_vec_t)] # use function to calculate intermediate values
    return (rho_vec_t,intermediate_data_list)
    
def MIS_remove_mat(): # matrix for removing element corresponding to two ones
    mat=np.eye(4)
    mat[3,3]=0
    return mat
    
def MIS_apply_phase(phi=np.pi): # matrix for appying a phase to enforce independence condition, defaults to a phase of -1
    mat=np.eye(4,dtype=complex)
    mat[3,3]=np.exp(1j*phi)
    return mat
    
def idealised_state_vector_anneal_wmis(Op_drive,phase_schedule,MIS_adj,driving='monolithic',driving_schedule=None,renormalize=False,interaction_Op=MIS_remove_mat(),phase_weights=None,intermediate_calc_functions=None): # applies an idealised anneal to solve a weighted maximum independent set in a state-vector representation
    '''
        Op_drive is a single subspace operator which drives the system, or the Hamiltonian which generates that operator (or a list of Hamiltonians each multiplied by -1j) if a driving schedule is supplied can be supplied as a list to give different driving on different modes
        phase_schedule is an iterable containing all the phases applied at different point
        MIS_adj is the adjacency matrix of a graph on which the independent set is to be found given as an array (only uses the upper triangular portion)
        driving determines how the driving is done, options are 'monolithic' for a single layer of driving, 'before' for interespered with interaction before each, 'after' for interspersed with interactions after each, and 'both' for interspersed with interaction both before and after
        driving_schedule is an optional schedule on the driving, if set to true, it assumes that the generators for the driving are provided as Op_drive rather than the driving operators themselves, currently ignored unless schedule is monolithic, if supplied as 'ideal' it trandforms the generators to only operate in the allowed subspace
        renormalization is a boolean which determines if the state vector should be renormalized each cycle
        phase_weights is an array of positive weights on each node, defaults to None in which case an unweighted problem is solved
        intermediate_calc_function is a function which calculates statistics for intermediate points in evolution, defaults to None, in which case nothing is applied
    '''
    num_qubits=MIS_adj.shape[0] # the number of qubit subspaces to be used
    (pos_inds_list,neg_inds_list)=phase_Op_0_1_pos_and_neg_inds(num_qubits) # pre-computed diagonal indices to make inner loop faster
    psi_t=np.zeros(2**num_qubits)
    psi_t[0]=1 # initialise as vacuum
    if driving=='ideal': # idealised version with perfect constraints
        forbid_mat=make_forbid_mat_MIS(MIS_adj) # matrix of forbidden values to apply to generator
    else:
        forbid_mat=None
    if not type(Op_drive)==type([]): # if a single operator rather than a list of operators
        drive_layer_Ops=single_mode_Op_layer([Op_drive]*num_qubits,factorised=True,forbid_mat=forbid_mat) # make an operator for the driving layer
    else:
        drive_layer_Ops=single_mode_Op_layer(Op_drive,factorised=True,forbid_mat=forbid_mat) # make an operator for driving based on supplied list
    indices_list=[]
    for i_MIS in range(num_qubits): # first adjacency index
        for j_MIS in range(i_MIS+1,num_qubits): # second adjacency index
            if not MIS_adj[i_MIS,j_MIS]==0: # if a non-zero entry in the adjacency matrix
                indices_list=indices_list+[[i_MIS,j_MIS]] # add to list of indices to create network to apply independence condition
    if driving=='monolithic': # driving done separately in monolithic case
        ind_layer=construct_interaction_Op_network_from_mat(interaction_Op,2,num_qubits,indices_list)
    else: # otherwise layers are combined
        if not type(Op_drive)==type([]): # if a single operator rather than a list of operators
            Op_drive=construct_normalised_drive_gen_list_Op(MIS_adj,Op_drive,driving_sequence=driving)
        if type(driving_schedule)==type(None):
            gen_dict=construct_gen_dict_drive_interaction_Op(interaction_Op,Op_drive,2,num_qubits,indices_list,forbid_mat=forbid_mat)
            ind_layer=construct_drive_interaction_Op_network_from_gen_dict(gen_dict,1,driving_sequence=driving)
            #ind_layer=construct_drive_interaction_Op_network_from_mat(interaction_Op,drive_layer_Ops,2,num_qubits,indices_list,driving_sequence=driving) # if no schedule make layer once
        else:
            gen_dict=construct_gen_dict_drive_interaction_Op(interaction_Op,Op_drive,2,num_qubits,indices_list,forbid_mat=forbid_mat) # otherwise produce a dictionary for final generation
    intermediate_data_list=[] # empty list for intermediate data
    for iStage in range(len(phase_schedule)): # each stage of evolution
        if type(phase_weights)==type(None): # if undefined, then create array of equal weights
            phase_layer=build_phase_layer_Op_0_1_from_indices(phase_schedule[iStage],num_qubits,pos_inds_list,neg_inds_list) # construct phase layer using pre-computed
        else: # otherwise calculate an individual superoperator for each mode
            phase_layer=build_phase_layer_Op_0_1_from_indices(phase_schedule[iStage]*phase_weights,num_qubits,pos_inds_list,neg_inds_list) # construct phase layer using pre-computed indices to avoid tensor products in the inner loop
        if driving=='monolithic': # driving done separately in monolithic case
            for drive_Op in drive_layer_Ops: # multiply one-by-one to avoid dense matrices
                if type(driving_schedule)==type(None): # if no driving schedule is provided
                    psi_t=drive_Op@psi_t # apply driving within one qubit subspace
                else:
                    psi_t=linalg_d.expm(drive_Op*driving_schedule[iStage])@psi_t # apply driving within one qubit subspace
        else:
            if (not type(driving_schedule)==type(None)): # final construction within inner loop if driving varied over time
                ind_layer=construct_drive_interaction_Op_network_from_gen_dict(gen_dict,driving_schedule[iStage],driving_sequence=driving)
        psi_t=phase_layer@psi_t # apply phase layer (diagonal so we can apply all at once)
        psi_t=ind_layer@psi_t # apply layer to enforce independence (sparse so we can apply all at once)
        if renormalize: # if we are renormalizing the state vector
            norm=psi_t@psi_t.conj().T # calculate the norm
            psi_t=psi_t/np.sqrt(norm) # renormalize state vector
        if not type(intermediate_calc_functions)==type(None):  # if intermediate values are to be computed
            intermediate_data_list=intermediate_data_list+[intermediate_calc_functions(psi_t)] # use function to calculate intermediate values
    return (psi_t,intermediate_data_list)
