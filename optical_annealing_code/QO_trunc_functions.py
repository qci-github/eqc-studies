import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import time
import scipy.optimize as opt
import scipy.special as special
import copy

def make_annihiltion_Op(n_trunc=100): # creates annihilation operator
    '''
        n_trunc is the maximum number of photons allowed in the mode, because vacuum is included, the size of the matrix is n_trunc+1
        this function returns a matrix reprsentation of the truncated annihilation operator for the mode returned as a sparse csr array
    '''
    data_list=[]
    row_ind_list=[]
    col_ind_list=[]
    for i_num in range(1,n_trunc+1):
       data_list=data_list+[np.sqrt(i_num)] # matrix elements
       row_ind_list=row_ind_list+[i_num-1] # row index
       col_ind_list=col_ind_list+[i_num] # column index
    a=sparse.csr_array((np.array(data_list),(np.array(row_ind_list),np.array(col_ind_list))),shape=(n_trunc+1,n_trunc+1)) # build sparse matrix
    return a
    
def get_subspace_photons(psi,n_before,n_after): # function for finding the maximum number of photons with a subspace of the tensor product
    '''
        psi is the state vector or integer size of a state vector
        n_before is the total number of modes appearing earlier in the tensor product
        n_after is the total number appearing after
        returns the maximum number of photons in the subspace (which is one less than the size of the matrix)
    '''
    if not type(psi)==int:
        tot_size=len(psi) # total size
    else:
        tot_size=psi # if it is size 1 assume it is the lenght of the vector rather than the actual vector
    sub_size=tot_size/(n_before*n_after) # divide to find subspace size
    if abs(sub_size-round(sub_size))>(10**-10): # if the size is not an integer to within numerical rounding error
        raise RuntimeError("non integer subspace size detected")
    else:
        return int(round(sub_size-1)) # round and convert to integer

def tensor_identities(Op,n_before,n_after): # tensor appropriately sized identity matrices before and after
    '''
        Op is an operator
        n_before is the number of matrix elements tensored before
        n_after is the number tensored after
        (note that before and after assume the first digit is the least significant bit)
    '''
    if sparse.issparse(Op): # if it is a sparse matrix
        Op_full=sparse.kron(Op,sparse.eye_array(n_before)) # tensor in values before
        Op_full=sparse.kron(sparse.eye_array(n_after),Op_full) # tensor in values after
    else:
        Op_full=np.kron(Op,np.eye(n_before)) # tensor in values before
        Op_full=np.kron(np.eye(n_after),Op_full) # tensor in values after
    return Op_full
    
    
def apply_time_evolution_Op(omegat,psi,n_before=1,n_after=1,gen_mat=False): # applies a displacement operator on a state vector
    '''
        omegat is a real parameter which defines the frequency times time which is applied following the definition of time evolution given in equation 1.108 of Gerry and Knight 
        psi is the state vector describing the system (or the size of the state vector if gen_mat==True)
        n_before and n_after are tensor product subsaces either before or after in the tensor product (default to one which assumes only a single state)
        gen_mat is a flag which asks the function to return the matrix which generates the time evolution rather than the state vector (to avoid overhead from repeated tensor products)
        returns the final wavefunction if gen_mat==False and the generating sparse matrix if gen_mat==True
    '''
    n_trunc_sub=get_subspace_photons(psi,n_before,n_after) # calculate the maximum number of photons in the subspace
    a_sub=make_annihiltion_Op(n_trunc=n_trunc_sub) # make the correctly sized annihilation operator
    a=tensor_identities(a_sub,n_before,n_after) # identities for unaffected modes as appropriate
    a_d=a.T.conj() # creation operator
    H_evo=a_d@a+sparse.eye_array(a.shape[0]) # create time evolution Hamiltonian
    if gen_mat:
        return -1j*omegat*H_evo # return matrix used in calculating evolution
    psi_f=linalg.expm_multiply(-1j*omegat*H_evo,psi) # final state vector after displacement
    return psi_f
    
    
def apply_displacement_Op(alpha,psi,n_before=1,n_after=1,gen_mat=False): # applies a displacement operator on a state vector
    '''
        alpha is a complex parameter which defines the displacement which is applied following the definition of displacement given in equation 3.30 of Gerry and Knight 
        psi is the state vector describing the system (or the size of the state vector if gen_mat=True)
        n_before and n_after are tensor product subsaces either before or after in the tensor product (default to one which assumes only a single state)
        returns the final wavefunction if gen_mat==False and the generating sparse matrix if gen_mat==True
    '''
    n_trunc_sub=get_subspace_photons(psi,n_before,n_after) # calculate the maximum number of photons in the subspace
    a_sub=make_annihiltion_Op(n_trunc=n_trunc_sub) # make the correctly sized annihilation operator
    a=tensor_identities(a_sub,n_before,n_after) # identities for unaffected modes as appropriate
    a_d=a.T.conj() # creation operator
    H_disp=alpha*a_d-np.conj(alpha)*a # create displacement Hamiltonian
    if gen_mat:
        return H_disp # return matrix used in calculating evolution
    psi_f=linalg.expm_multiply(H_disp,psi) # final state vector after displacement
    return psi_f
    
def apply_squeezing_Op(xi,psi,n_before=1,n_after=1,gen_mat=False): # applies a squeezing operator on a state vector
    '''
        xi is a complex parameter which defines the displacement which is applied following the definition of squeezing given in equation 7.10 of Gerry and Knight 
        psi is the state vector describing the system (or the size of the state vector if gen_mat==True)
        n_before and n_after are tensor product subsaces either before or after in the tensor product (default to one which assumes only a single state)
        returns the final wavefunction if gen_mat==False and the generating sparse matrix if gen_mat==True
    '''
    n_trunc_sub=get_subspace_photons(psi,n_before,n_after) # calculate the maximum number of photons in the subspace
    a_sub=make_annihiltion_Op(n_trunc=n_trunc_sub) # make the correctly sized annihilation operator
    a=tensor_identities(a_sub,n_before,n_after) # identities for unaffected modes as appropriate
    a_d=a.T.conj() # creation operator
    H_sq=0.5*(np.conj(xi)*a@a-xi*a_d@a_d) # create squeezing Hamiltonian
    if gen_mat:
        return H_sq # return matrix used in calculating evolution
    psi_f=linalg.expm_multiply(H_sq,psi) # final state vector after squeezing
    return psi_f
    
def apply_beamsplitting_Op(theta,psi,n_before,n_after,gen_mat=False): # applies a beamsplitting operator on a state vector
    '''
        theta is a real parameter which defines the degree of beamsplitting which is applied following an extension of the definition of beamsplitting given in equation 6.12 of Gerry and Knight such that theta=pi/4 corresponds to a 50:50 beamsplitter 
        psi is the state vector describing the system (or the size of the state vector if gen_mat=True)
        n_before and n_after are lists of the number before and after for each mode such that n_before[0] and n_after[0] correspond to the first mode and n_before[1] and n_after[1] correspond to the second mode
        returns the final wavefunction if gen_mat==False and the generating sparse matrix if gen_mat==True
    '''
    n_trunc_sub_0=get_subspace_photons(psi,n_before[0],n_after[0]) # calculate the maximum number of photons in the first subspace
    a_sub_0=make_annihiltion_Op(n_trunc=n_trunc_sub_0) # make the correctly sized annihilation operator
    a_0=tensor_identities(a_sub_0,n_before[0],n_after[0]) # identities for unaffected modes as appropriate
    a_0_d=a_0.T.conj() # creation operator
    n_trunc_sub_1=get_subspace_photons(psi,n_before[1],n_after[1]) # calculate the maximum number of photons in the second subspace
    a_sub_1=make_annihiltion_Op(n_trunc=n_trunc_sub_1) # make the correctly sized annihilation operator
    a_1=tensor_identities(a_sub_1,n_before[1],n_after[1]) # identities for unaffected modes as appropriate
    a_1_d=a_1.T.conj() # creation operator
    H_bs=1j*theta*(a_0_d@a_1+a_0@a_1_d) # create beamspliting Hamiltonian
    if gen_mat:
        return H_bs # return matrix used in calculating evolution
    psi_f=linalg.expm_multiply(H_bs,psi) # final state vector after beamsplitting
    return psi_f
    
def apply_chi_2_deg_Op(theta,psi,n_before,n_after,gen_mat=False): # applies a beamsplitting operator on a state vector
    '''
        theta is a real parameter which defines the degree of nonlinear application theta=np.pi/(2*np.sqrt(2)) correspondst to complete transfer to the pump mode
        psi is the state vector describing the system (or the size of the state vector if gen_mat=True)
        n_before and n_after are lists of the number before and after for each mode such that n_before[0] and n_after[0] correspond to the first mode and n_before[1] and n_after[1] correspond to the second mode
        returns the final wavefunction if gen_mat==False and the generating sparse matrix if gen_mat==True
    '''
    n_trunc_sub_0=get_subspace_photons(psi,n_before[0],n_after[0]) # calculate the maximum number of photons in the first subspace
    a_sub_0=make_annihiltion_Op(n_trunc=n_trunc_sub_0) # make the correctly sized annihilation operator
    a_0=tensor_identities(a_sub_0,n_before[0],n_after[0]) # identities for unaffected modes as appropriate
    a_0_d=a_0.T.conj() # creation operator
    n_trunc_sub_1=get_subspace_photons(psi,n_before[1],n_after[1]) # calculate the maximum number of photons in the second subspace
    a_sub_1=make_annihiltion_Op(n_trunc=n_trunc_sub_1) # make the correctly sized annihilation operator
    a_1=tensor_identities(a_sub_1,n_before[1],n_after[1]) # identities for unaffected modes as appropriate
    a_1_d=a_1.T.conj() # creation operator
    H_nl=1j*theta*(a_0_d@a_0_d@a_1+a_0@a_0@a_1_d) # create nonlinear Hamiltonian
    if gen_mat:
        return H_nl # return matrix used in calculating evolution
    psi_f=linalg.expm_multiply(H_nl,psi) # final state vector after beamsplitting
    return psi_f
    
def make_sparse_project(n_proj,n_sub): # creates an 1-by-n sparse matrix which can be used to project out a single value
    '''
        n_proj is the number to be projected out
        n_sub is the number of elements in the subspace to be projected
    '''
    data_list=[1]
    row_ind_list=[0]
    col_ind_list=[n_proj]
    p=sparse.csr_array((np.array(data_list),(np.array(row_ind_list),np.array(col_ind_list))),shape=(1,n_sub)) # build sparse matrix
    return p

def prob_proj_subspace(n_sub): # creates an 1-by-n sparse matrix which can be used to project out a single value
    '''
        n_sub is the number of elements in the subspace to be projected
    '''
    data_list=[1]*n_sub
    row_ind_list=[0]*n_sub
    col_ind_list=[i for i in range(n_sub)]
    p=sparse.csr_array((np.array(data_list),(np.array(row_ind_list),np.array(col_ind_list))),shape=(1,n_sub)) # build sparse matrix
    return p

    
def photon_number_subspace_project(n_proj,n_sub,n_before=1,n_after=1): # projects into a given number of photons in a given mode
    '''
        n_proj is the number of photons in the state to be projected
        n_sub is the total size of the subspace (number of allowed photons plus 1) 
        n_before is the number of total elements tensored before
        n_after is the number of total elements tensored after
    
    '''
    p=make_sparse_project(n_proj,n_sub)
    p=sparse.kron(p,sparse.eye_array(n_before)) # tensor in values before
    p=sparse.kron(sparse.eye_array(n_after),p) # tensor in values after
    return p
    
def photon_number_prob_sum(n_sub,n_before=1,n_after=1): # projects into a given number of photons in a given mode
    '''
        n_sub is the total size of the subspace to be summed over (number of allowed photons plus 1) 
        n_before is the number of total elements tensored before
        n_after is the number of total elements tensored after
    
    '''
    p=prob_proj_subspace(n_sub)
    p=sparse.kron(p,sparse.eye_array(n_before)) # tensor in values before
    p=sparse.kron(sparse.eye_array(n_after),p) # tensor in values after
    return p


def photon_number_subspace_probabilities(psi,n_sub,n_before=1,n_after=1,gen_mats=False): # calculates the probability of having a given number of photons in a given mode
    '''
        psi is the state vector (not used if gen_mat=True)
        n_sub is the total size of the subspace (number of allowed photons plus 1) 
        n_before is the number of total elements tensored before
        n_after is the number of total elements tensored after
        gen_mat is a flag which only gives the projector matrices if set to True
    '''
    if (n_before==1 and n_after==1): # if there are no subspaces then the probabilities are just amplitude squared
        probs=np.abs(psi)**2
    else:
        if gen_mats: # if we are only intersted in the projection matrices
            proj_mats=[] # empty list
        else:
            probs=np.zeros(n_sub) # array of zeros for probabilities
        for n_proj in range(n_sub):
            p=photon_number_subspace_project(n_proj,n_sub,n_before=n_before,n_after=n_after)
            if gen_mats:
                proj_mats=proj_mats+[p]
            else:
                psi_proj=p@psi
                probs[n_proj]=sum(abs(psi_proj)**2)
    if gen_mats:
        return proj_mats
    return probs
    
def mean_photon_count_subspace(psi,n_sub,n_before=1,n_after=1): # counts the number of photons in a subspace
    '''
        psi is the state vector
        n_sub is the total size of the subspace (number of allowed photons plus 1) 
        n_before is the number of total elements tensored before
        n_after is the number of total elements tensored after
    '''
    probs=photon_number_subspace_probabilities(psi,n_sub,n_before=n_before,n_after=n_after) # calculate probabilities
    photon_count=0 # keep track of total count
    for n_photon in range(n_sub): # loop through number of photons in each state
        photon_count=photon_count+probs[n_photon]*n_photon # zero based indexing directly maps to photon number
    return photon_count
    
def make_X1_X2_ops(n_sub,n_before=1,n_after=1): # makes the quadrature operators for a subspace for size n_sub
    '''
        n_sub is the total size of the subspace (number of allowed photons plus 1) 
        returns an X1 and X2 operator following equations 2.52 and 2.53 of Gerry and Knight
        n_before is the number of total elements tensored before
        n_after is the number of total elements tensored after
    '''
    a=tensor_identities(make_annihiltion_Op(n_sub-1),n_before,n_after) # annihilation operator
    a_d=a.T.conj() # creation operator
    X1=0.5*(a+a_d) # first quadrature
    X2=-0.5*1j*(a-a_d) # second quadrature
    return (X1,X2) # return quadrature operators
    

    
def calculate_Q_pure_state(alpha,psi): # calculates a single value of the Q function at dispacement alpha
    '''
        alpha is the displacement value
        psi is the wavefunction against which the Q value is calculated
        based on definition in Gerry and Knight Eq. 3.112
    '''
    psi_comp=np.zeros(len(psi)) # zero vector
    psi_comp[0]=1 # vacuum state
    psi_comp=apply_displacement_Op(alpha,psi_comp) # apply displacement
    Q=(1/np.pi)*abs(psi@psi_comp)**2 # squared fidelity
    return Q
    


def make_Q_plot_data_pure_state(psi,re_lims=[-10,10],im_lims=[-10,10],point_resolution=51): # makes the data for a plot of the Husimi Q function
    '''
        psi is the wavefunction for which the plot is being made
        re_lims is the limits on the real values of alpha as a list with two entries, defaults to between -10 and 10
        im_lims is the limits on the imaginary values fo alpha for the plot defaults to between -10 and 10
        point_resolution is how many points to take in each direction (produces an array of size point_resolution**2) defaults to 51 so that the values of exactly zero for real and imaginary parts are captured (low value for fast plotting 201 is probably better for production plots)
    '''
    re_alpha_array=np.linspace(re_lims[0],re_lims[1],point_resolution) # array of real parts of alpha values
    im_alpha_array=np.linspace(im_lims[0],im_lims[1],point_resolution) # array of imaginary parts of alpha values
    Q_array=np.zeros([point_resolution,point_resolution]) # array for Q values
    for i_re in range(point_resolution): # real part of alpha
        for i_im in range(point_resolution): # imaginary part of alpha
            Q_array[i_re,i_im]=calculate_Q_pure_state(re_alpha_array[i_re]+1j*im_alpha_array[i_im],psi) # calculate Q value and add to array
    return (re_alpha_array,im_alpha_array,Q_array)
    
def calculate_pair_Q_pure_state(alpha1,alpha2,psi,n_sub_1=None): # calculates a two variable generalisation of the Q function for alpha1 and alpha2
    '''
        alpha1 is the first displacement value
        alpha2 is the second displacement value
        psi is the wavefunction
        n_sub_1 is the size of the first subspace, if not supplied it is assumed to be the square root of the length of psi
    '''
    if type(n_sub_1)==type(None): # if a value has not been supplied
        n_sub_1=int(round(np.sqrt(len(psi)))) # assume subspaces are of equal size
    n_sub_2=int(round(len(psi)/n_sub_1)) # infer size of second subspace
    psi_comp=np.zeros(len(psi)) # zero vector
    psi_comp[0]=1 # vacuum state
    psi_comp=apply_displacement_Op(alpha1,psi_comp,n_after=n_sub_2) # apply displacement to first subspace
    psi_comp=apply_displacement_Op(alpha2,psi_comp,n_before=n_sub_1) # apply displacement to second subspace
    Q=abs(psi@psi_comp)**2 # squared fidelity
    return Q
    
def make_Q_pair_plot_data_pure_state(psi,alpha1_lims=[-5,5],alpha2_lims=[-5,5],point_resolution=51,phi_alpha_1=0,phi_alpha_2=0,n_sub_1=None): # makes the data for a plot of the two variable generalisation of the Q function
    '''
        psi is the wavefunction for which the plot is being made
        alpha2_lims is the limits on the values of the first alpha, defaults to between -5 and 5
        alpha2_lims is the limits on the values of the first alpha, defaults to between -5 and 5
        point_resolution is how many points to take in each direction (produces an array of size point_resolution**2) defaults to 51 so that the values of exactly zero for real and imaginary parts are captured (low value for fast plotting 201 is probably better for production plots)
        phi_alpha_1 is the phase to apply to alpha1
        phi_alpha_2 is the phase to apply to alpha2
        n_sub_1 is the size of the first subspace, if not supplied it is assumed to be the square root of the length of psi
    '''
    alpha1_array=np.linspace(alpha1_lims[0],alpha1_lims[1],point_resolution)*np.exp(1j*phi_alpha_1) # first alpha value array
    alpha2_array=np.linspace(alpha1_lims[0],alpha2_lims[1],point_resolution)*np.exp(1j*phi_alpha_2) # second alpha value array
    Q_array=np.zeros([point_resolution,point_resolution]) # array for Q values
    for i_1 in range(point_resolution): # alpha1
        for i_2 in range(point_resolution): # alpha2
            Q_array[i_1,i_2]=calculate_pair_Q_pure_state(alpha1_array[i_1],alpha2_array[i_2],psi,n_sub_1=n_sub_1) # calculate Q value and add to array
    return (alpha1_array,alpha2_array,Q_array)

def calculate_displacement_and_covariance_pure_state(psi,n_sub_1=None): # calculates the displacement and covariance matrix of a pure state over two modes
    '''
        psi is the two mode wavefunction
        n_sub_1 is the size of the first subspace (the number of photons allowed plus 1), defaults to being the square root of the length of psi
        returns the displacement d and the covariance sigma as defined in arXiv:2102.05748v2
    '''
    if type(n_sub_1)==type(None): # if a value has not been supplied
        n_sub_1=int(round(np.sqrt(len(psi)))) # assume subspaces are of equal size
    n_sub_2=int(round(len(psi)/n_sub_1)) # infer size of second subspace
    Quad_list=[None]*4 # empty list for quadrature operators
    (Quad_list[0],Quad_list[1])=make_X1_X2_ops(n_sub_1,n_after=n_sub_2) # quadrature operators in first subspace
    (Quad_list[2],Quad_list[3])=make_X1_X2_ops(n_sub_2,n_before=n_sub_1) # quadrature operators in second subspace
    d_arr=np.zeros(4)
    for i_d in range(4): # loop through quadrature operators
        d_arr[i_d]=np.real(psi.conj().T@Quad_list[i_d]@psi) # inner product to calculate quadrature
    sigma_arr=np.zeros([4,4])
    for i_cov in range(4): # first index for finding covanriances
        for j_cov in range(i_cov,4): # second index for finding covanriances
            sigma_arr[i_cov,j_cov]=2*np.real((psi.conj().T@Quad_list[i_cov]@Quad_list[j_cov]@psi+psi.conj().T@Quad_list[j_cov]@Quad_list[i_cov]@psi)/2-d_arr[i_cov]*d_arr[j_cov]) # calculate by equation 15 in arXiv:2102.05748v2 note factor of sqrt(2) difference in p and q from G+K X1 and X2 definition
            sigma_arr[j_cov,i_cov]=sigma_arr[i_cov,j_cov] # symmetric by definition
    return (d_arr,sigma_arr)
    
def rotate_covariance_mat_two_mode(sigma_arr,theta_1=0,theta_2=0): # perform phase rotations on a covariance matrix describing two modes
    '''
        sigma_arr is a 4x4 array describing the covariance of the two modes as defined in arXiv:2102.05748v2
        performs rotations of covariance matrix using F as defined in equation 33 and 35 of arXiv:2102.05748v2
    '''
    F=np.zeros([4,4]) # empty array to start with
    F[0,0]=np.cos(theta_1)
    F[1,1]=np.cos(theta_1)
    F[1,0]=-np.sin(theta_1)
    F[0,1]=np.sin(theta_1)
    F[2,2]=np.cos(theta_2)
    F[3,3]=np.cos(theta_2)
    F[3,2]=-np.sin(theta_2)
    F[2,3]=np.sin(theta_2)
    sigma_arr_new=F@sigma_arr@F.T # equation 23 of arXiv:2102.05748v2
    return sigma_arr_new
    
def beamsplit_covariance(sigma_arr,theta): # perform a beamsplitting operation on a covariance matrix describing two modes
    '''
        sigma_arr is a 4x4 array describing the covariance of the two modes as defined in arXiv:2102.05748v2
        theta is the beamsplitting angle, in the notation of arXiv:2102.05748v2 eta=cos(theta)**2
        performs beamsplitting as defined in equation 41 of arXiv:2102.05748v2
    '''
    F=np.kron([[0,1],[-1,0]],np.eye(2))*np.sin(theta)+np.eye(4)*np.cos(theta)
    sigma_arr_new=F@sigma_arr@F.T # equation 23 of arXiv:2102.05748v2
    return sigma_arr_new
    
def cross_covariance_measure(sigma_arr): # defines a measure of the cross-covarianance between two modes, defined as trace of the square of the matrix with the diagonal blocks removed
    '''
        sigma_arr is a 4x4 array describing the covariance of the two modes as defined in arXiv:2102.05748v2
        returns measure of cross terms in covariance matrix
    '''
    sigma_arr_new=np.zeros([4,4]) # empty 4x4 array
    sigma_arr_new[0:2,2:4]=copy.copy(sigma_arr[0:2,2:4]) # cross-mode elements only
    sigma_arr_new[2:4,0:2]=copy.copy(sigma_arr[2:4,0:2]) # cross-mode elements only
    cross_cov=np.sum(np.diag(sigma_arr_new@sigma_arr_new)) # square the matrix and sum the diagonal elements
    return cross_cov
    
def cross_cov_objective(x,sigma_arr): # function for calculating objective in removing cross covariance
    '''
        x is a two element vector of the rotation angle on the second mode and the beamsplitting angle, which are optimised over
        sigma_arr is the initial 4x4 covariance matrix for the modes
    '''
    sigma_arr_1=rotate_covariance_mat_two_mode(sigma_arr,theta_1=0,theta_2=x[0]) # initial rotation of second mode
    sigma_arr_1=beamsplit_covariance(sigma_arr_1,theta=x[1]) # beamsplitting
    obj=cross_covariance_measure(sigma_arr_1) # calculate objective value
    return obj
    
def remove_cross_cov_params(sigma_arr,n_theta_rot=20,n_theta_bs=20): # use scipy optimize to find rotation and beamsplitting angle to remove cross covariace in two mode system
    '''
        sigma_arr is the initial 4x4 covariance matrix for the modes
        n_theta_rot is the number of rotation angles to use in an initial grid search
        n_theta_bs is the number of rotation angles to use in an initial grid search
        returns optimal roation and beamsplitting angles
    '''
    obj_best=np.inf # set to be infinite so that any finite value will replace it
    theta_rot_list=list(np.linspace(0,np.pi,n_theta_rot)) # list of rotation angles
    theta_bs_list=list(np.linspace(0,np.pi/4,n_theta_bs)) # list of beamsplitter angles
    for theta_rot_test in theta_rot_list: # test different rotation angles
        for theta_bs_test in theta_bs_list: # test different beamsplitting angles
            obj_test=cross_cov_objective([theta_rot_test,theta_bs_test],sigma_arr) # calculate objective value
            if obj_test<obj_best: # if less than best value
                obj_best=copy.copy(obj_test)
                theta_rot_best=copy.copy(theta_rot_test)
                theta_bs_best=copy.copy(theta_bs_test)
    f_opt=lambda x: cross_cov_objective(x,sigma_arr)
    #bounds=[(0,np.pi),(0,np.pi/4)]
    optimum=opt.minimize(f_opt,[theta_rot_best,theta_bs_best],method='BFGS') # use scipy to improve values from grid search
    theta_rot_best=optimum.x[0]
    theta_bs_best=optimum.x[1]
    obj_best=cross_cov_objective([theta_rot_best,theta_bs_best],sigma_arr)
    return(theta_rot_best,theta_bs_best,obj_best)
    


def remove_cross_cov(psi,n_sub_1=None): # function for removing cross covariance from a wavefuntion
    '''
        psi is the two mode wavefunction
        n_sub_1 is the size of the first subspace (the number of photons allowed plus 1), defaults to being the square root of the length of psi
        returns the wavefunction of the same system but with rotation and beamsplitting applied to remove cross covariance
    '''
    (d_arr,sigma_arr)=calculate_displacement_and_covariance_pure_state(psi,n_sub_1=n_sub_1) # calculate covariance matrix
    (theta_rot_best,theta_bs_best,obj_best)=remove_cross_cov_params(sigma_arr) # find best beamsplitting and rotation angles to remove cross-covariances
    if type(n_sub_1)==type(None): # if a value has not been supplied
        n_sub_1=int(round(np.sqrt(len(psi)))) # assume subspaces are of equal size
    n_sub_2=int(round(len(psi)/n_sub_1)) # infer size of second subspace
    psi_1=apply_time_evolution_Op(theta_rot_best+np.pi/2,psi,n_before=n_sub_1) # apply rotation
    psi_1=apply_beamsplitting_Op(theta_bs_best,psi_1,n_before=[1,n_sub_1],n_after=[n_sub_2,1]) # apply beamsplitting
    return psi_1
    
    
    
def find_squeezing_single_mode(sigma_arr): # finds the value of the complex parameter xi to give a single mode with the same covariance matrix
    '''
        sigma_arr is a single mode 2x2 covaraiance matrix as defined in arXiv:2102.05748v2
        returns two real parameters r and theta such that xi=r*np.exp(1j*theta) as defined in arXiv:2102.05748v2
    '''
    z_component=(sigma_arr[0,0]-sigma_arr[1,1])/2 # Pauli-Z component is half the difference in the diagonal entries
    x_component=sigma_arr[0,1] # Pauli-X component is the off-diagonal entry
    id_component=(sigma_arr[0,0]+sigma_arr[1,1])/2 # identity matrix component is half the sum in the diagonal entries
    theta=np.arctan(x_component/z_component) # angle of squeezing up to a factor of pi
    if (theta>0 and x_component>0) or (theta<0 and x_component<0) or (x_component==0 and z_component>0): # if the x-component is positive or exactly zero but the z component is positive
        theta=theta+np.pi # apply a pi phase shift
    r=np.arctanh(np.sqrt(z_component**2+x_component**2)/(id_component))/2 # magnitude of squeezing, factor of two because this is covarianance rather than Delta X
    return (r,theta) # return squeezing magnitude and angle

def two_mode_displacement_reversal(psi,n_sub_1=None): # takes a two mode system and removes displacement
    '''
        psi is the two mode wavefunction
        n_sub_1 is the size of the first subspace (the number of photons allowed plus 1), defaults to being the square root of the length of psi
        returns the wavefunction of the same system but with the diplacement set to zero in each mode
    '''
    if type(n_sub_1)==type(None): # if a value has not been supplied
        n_sub_1=int(round(np.sqrt(len(psi)))) # assume subspaces are of equal size
    n_sub_2=int(round(len(psi)/n_sub_1)) # infer size of second subspace
    (d_arr,sigma_arr)=calculate_displacement_and_covariance_pure_state(psi,n_sub_1=n_sub_1) # calculate displacement and covariance
    psi_nd=apply_displacement_Op(-d_arr[0]-1j*d_arr[1],psi,n_before=1,n_after=n_sub_2) # remove displacement on first mode
    psi_nd=apply_displacement_Op(-d_arr[2]-1j*d_arr[3],psi_nd,n_before=n_sub_1,n_after=1) # remove diplacement on second mode
    return psi_nd # return the the wavefunction with displacement removed
    
def two_mode_squeezing_reversal(psi,n_sub_1=None): # takes a two mode system and removes squezing from each individual mode
    '''
        psi is the two mode wavefunction
        n_sub_1 is the size of the first subspace (the number of photons allowed plus 1), defaults to being the square root of the length of psi
        returns the wavefunction of the same system but with the diplacement set to zero in each mode
    '''
    if type(n_sub_1)==type(None): # if a value has not been supplied
        n_sub_1=int(round(np.sqrt(len(psi)))) # assume subspaces are of equal size
    n_sub_2=int(round(len(psi)/n_sub_1)) # infer size of second subspace
    (d_arr,sigma_arr)=calculate_displacement_and_covariance_pure_state(psi,n_sub_1=n_sub_1) # calculate displacement and covariance
    (r_1,theta_1)=find_squeezing_single_mode(sigma_arr[0:2,0:2]) # squeezing parameters for first mode
    if not np.isnan(theta_1): # for the case where there is no squeezing
        psi_ns=apply_squeezing_Op(-r_1*np.exp(1j*theta_1),psi,n_before=1,n_after=n_sub_2) # remove displacement on first mode
    else:
        psi_ns=psi
    (r_2,theta_2)=find_squeezing_single_mode(sigma_arr[2:4,2:4]) # squeezing parameters for second mode
    if not np.isnan(theta_2): # for the case where there is no squeezing
        psi_ns=apply_squeezing_Op(-r_2*np.exp(1j*theta_2),psi_ns,n_before=n_sub_1,n_after=1) # remove squeezing on second mode
    return psi_ns # return the the wavefunction with displacement removed
    
    
def unmake_2mode_Gaussian_state(psi,n_sub_1=None,squeeze_removals=1): # performs a procedure which can turn any two-mode Gaussian state into a tensor of two vacuum states, remaining non-vacuum amplitude represents non-Gaussian parts of the state
    '''
        psi is the two mode wavefunction
        n_sub_1 is the size of the first subspace (the number of photons allowed plus 1), defaults to being the square root of the length of psi, set to 1 for single mode operation
        returns the wavefunction after a unitary transformation to remove the Gaussian degrees of freedom
    '''
    if type(n_sub_1)==type(None): # if a value has not been supplied
        n_sub_1=int(round(np.sqrt(len(psi)))) # assume subspaces are of equal size
    if False: #n_sub_1>1: # setting n_sub_1 to 1 allows this function to unmake a single mode
        psi_1=remove_cross_cov(psi,n_sub_1=n_sub_1) # remove the cross-covariance from each mode
    else:
        psi_1=copy.copy(psi)
    psi_1=two_mode_displacement_reversal(psi_1,n_sub_1=n_sub_1) # reverse the displacement on each mode
    if n_sub_1>1:
        psi_1=remove_cross_cov(psi_1,n_sub_1=n_sub_1) # remove the cross-covariance from each mode
    for i_sq in range(squeeze_removals):
        psi_1=two_mode_squeezing_reversal(psi_1,n_sub_1=n_sub_1) # undo squeezing on the two modes
    return psi_1
    
def non_Gaussianity_measures_2mode(psi,n_sub_1=None,squeeze_removals=1): # measures overlap with Gaussian state and fraction of photons in non-Gaussian degrees of freedom
    if type(n_sub_1)==type(None): # if a value has not been supplied
        n_sub_1=int(round(np.sqrt(len(psi)))) # assume subspaces are of equal size
    n_sub_2=int(round(len(psi)/n_sub_1)) # infer size of second subspace
    n_photon_orig=mean_photon_count_subspace(psi,n_sub_1,n_after=n_sub_2)
    n_photon_orig=n_photon_orig+mean_photon_count_subspace(psi,n_sub_2,n_before=n_sub_1)
    psi_um=unmake_2mode_Gaussian_state(psi,n_sub_1=n_sub_1,squeeze_removals=squeeze_removals)
    Gauss_fid=abs(psi_um[0])**2 # vaccuum probability after Gaussian unmaking procedure
    n_photon_ng=mean_photon_count_subspace(psi_um,n_sub_1,n_after=n_sub_2)
    n_photon_ng=n_photon_ng+mean_photon_count_subspace(psi_um,n_sub_2,n_before=n_sub_1)
    return (Gauss_fid,n_photon_ng,n_photon_orig)
    

def find_disp_and_squeezing_photon_number(n_hat_disp,n_hat_squeeze): # calculates the maginitude of alpha and xi for a given (average) number of photons from displacement and a given (average) number from squeezing (based on eq. 7.100 in Gerry and Knight)
    '''
        n_hat_disp is the (average) number of photons from displacement used to calculate the displacement magnitude based on based on eq. 7.100 in Gerry and Knight
        n_hat_squeeze is the (average) number of photons from squeezing used to calculate the squeezing magnitude based on based on eq. 7.100 in Gerry and Knight
    '''
    alpha_abs=np.sqrt(n_hat_disp) # magnitude of displacement
    xi_abs=np.arcsinh(np.sqrt(n_hat_squeeze)) # magnitude of squeezing
    return(alpha_abs,xi_abs)
    
    
def photon_number_diff_statistics(psi,n_sub_1=None): # calculate stats related to difference in the number of photons in two modes
    '''
        psi is a two-mode wavefunction
        n_sub_1 is the size of the first subspace (defaults to np.sqrt(len(psi) if set to None)
    '''
    if type(n_sub_1)==type(None): # if not supplied
        n_sub_1=int(np.sqrt(len(psi))) # half the total space
    n_sub_2=int(len(psi)/n_sub_1) # size of second subspace
    prob_arr=np.reshape(abs(psi)**2,[n_sub_1,n_sub_2])  # reshape the array of probabilities
    diff_arr=np.array(range(-n_sub_2,n_sub_1+1)) # array of photon number differences between subspace one and two
    diff_probs=np.zeros(len(diff_arr)) # array for probabilities of different differences
    for i_sub_1 in range(n_sub_1):
        for i_sub_2 in range(n_sub_2):
            prob_ind=i_sub_1-i_sub_2+n_sub_2 # index to add to difference probabilities
            diff_probs[prob_ind]=diff_probs[prob_ind]+prob_arr[i_sub_1,i_sub_2] # difference probabilities
    return(prob_arr,diff_arr,diff_probs)
    

    
def prepare_squeezed_state_photon_number(n_hat_disp=0,n_hat_squeeze=0,n_sub=31,phi_disp=0,phi_squeeze=0,start_photons=0,vacuum=False): # prepares a squeezed state with a given (average) number of photons from displacement and a given (average) number from squeezing (based on eq. 7.100 in Gerry and Knight)
    '''
        n_hat_disp is the (average) number of photons from displacement used to calculate the displacement magnitude defaults to zero
        n_hat_squeeze is the (average) number of photons from squeezing used to calculate the squeezing magnitude defaults to zero
        n_sub is the size of the subspace (maximum allowed photon number plus 1
        phi_disp is the phase of the displacement (defaults to zero)
        phi_squeeze is the phase of the squeezing (defaults to zero)
        start_photons is the number of photons initially in the system defaults to zero photons
        vacuum is a boolean which if set to true overrides all other parameters and inputs vacuum useful when assessing the effect of interference)
    '''
    (alpha_abs,xi_abs)=find_disp_and_squeezing_photon_number(n_hat_disp,n_hat_squeeze) # calcuate the magnitude of the squeezing and displacement
    psi=np.zeros(n_sub) # create vector of correct size
    if vacuum: # if we have overidden the other parameters
        psi[0]=1 # create vacuum
    else:
        psi[start_photons]=1 # start in a number state (defaults to vacuum)
        psi=apply_squeezing_Op(xi_abs*np.exp(1j*phi_squeeze),psi) # squeezing
        psi=apply_displacement_Op(alpha_abs*np.exp(1j*phi_disp),psi) # displacement
    return psi
    
def prepare_cat_state_photon_number(n_hat_disp=1,n_sub=31,phi_disp=np.pi/2,phi_sup=np.pi,n_hat_squeeze=0,phi_squeeze=0): # prepares a schrodinger cat state with a given (average) number of photons (based on eq. 7.100 in Gerry and Knight) with optional additional (anti-)squeezing photons
    '''
        n_hat_disp is the (average) number of photons from displacement used to calculate the displacement magnitude defaults to zero
        n_sub is the size of the subspace (maximum allowed photon number plus 1
        phi_disp is the phase of the displacement (defaults to zero)
        phi_sup is a complex number corresponding to the relative phase used in the superposition
    '''
    (alpha_abs,xi_abs)=find_disp_and_squeezing_photon_number(n_hat_disp,n_hat_squeeze=n_hat_squeeze) # calcuate the magnitude of the squeezing and displacement
    psi_p=prepare_squeezed_state_photon_number(n_hat_disp,n_sub=n_sub,phi_disp=phi_disp,n_hat_squeeze=n_hat_squeeze,phi_squeeze=phi_squeeze) # create positive phase state
    psi_m=prepare_squeezed_state_photon_number(n_hat_disp,n_sub=n_sub,phi_disp=phi_disp+np.pi,n_hat_squeeze=n_hat_squeeze,phi_squeeze=phi_squeeze) # create negative phase state
    psi=psi_p+np.exp(1j*phi_sup)*psi_m # sum with perscribed relative phase
    psi=psi/np.sqrt(sum(abs(psi)**2)) # normalize
    return psi
    
def squeeze_cat_fid(psi,phi_sup=np.pi,phi_disp=np.pi/2,phi_squeeze=0): # calculates the overlap with a cat-like state prepared with (anti-)squeezed vacuum
    '''
        psi is the state which is to be fit
        phi_sup is the angle of the superposition (defaults to np.pi)
        phi_squeeze is the angle of the squeezing (defaults to zero)
    '''
    n_sub=len(psi) # the size of the hilbert space
    n_hat_total=mean_photon_count_subspace(psi,n_sub) # average number of photons
    f_opt=lambda x: -abs(psi@prepare_cat_state_photon_number((1-x)*n_hat_total,n_sub=n_sub,phi_sup=phi_sup,n_hat_squeeze=x*n_hat_total,phi_squeeze=phi_squeeze,phi_disp=phi_disp))**2 # objective is to minimize the fidelity while varying the fraction of the photons which are from squeezing
    optimum=opt.minimize(f_opt,[0],bounds=[(0,1)]) # optmise starting with no photons from squeezing
    squeeze_frac_best=optimum.x[0] # fraction from squeezing in the best case
    abs_fid_best=-f_opt(squeeze_frac_best) # best absolute fidelity
    return(abs_fid_best,squeeze_frac_best,n_hat_total) # return best fidelity and information about the best state
    
    
def squeeze_beam_split_analytic(k,n_hat_squeeze=np.infty,theta=0,theta_subtract=np.pi/5,n_sub=1001,return_p=False): # calculates the result of beamsplitting a squeezed vacuum followed by number measurement from an analytical formula, applies beamsplitting to a squeezed state created using the recursive definition given in eq. 7.60 of G+K
    '''
        k is the measured number of photons
        n_hat_squeeze is the number of photons of squeezing added, used to back-calculate the amplitude of squeezing, r, can be sent to np.infty for infinite squeezing limit, this is the default behaviour
        theta is the squeeezing angle
        theta_subtract is the beamsplitting angle used when subtracting the photons
        n_sub is the truncated size of the final state vector (maximum number of photons plus one)
        return_p is a flag to return a factor proportional to the probability which can be used to calculate probs by normalising
    '''
    psi_out=np.zeros(n_sub) # initialize state as all zeros vector
    (alpha_abs,xi_abs)=find_disp_and_squeezing_photon_number(0,n_hat_squeeze)
    C_squeeze=1 # initialise the coefficiants for recursive calculation
    nu_over_mu=np.exp(1j*theta)*np.tanh(xi_abs) # definition of the ratio of these coefficients from G+K
    squeeze_norm_sq=0 # squared norm for squeezed state (used in calculating probability)
    for i_photon in range(k): # initial loop which doesn't add to psi but ensures the norm is correct
        if i_photon%2==0: # if i_photon is even
            squeeze_norm_sq=squeeze_norm_sq+C_squeeze**2 # add square of coefficient
            n_new=i_photon+1 # value of n to be used in 7.60 for next term in the recursion
            C_squeeze=-nu_over_mu*np.sqrt(n_new/(n_new+1))*C_squeeze # apply recursion relationship from 7.60
    for i_photon in range(n_sub): # loop through number of photons in the final state
        n=i_photon+k
        if n%2==0: # if n is even
            beamsplit_prefactor=np.sqrt(special.comb(n,k))*(np.cos(theta_subtract)**i_photon) # un-normalised prefactor for beamsplitting
            #C_squeeze=((-1)**(n/2))*(np.sqrt(special.factorial(n))/((2**(n/2))*special.factorial(n/2)))*np.exp(1j*theta*n/2)*np.tanh(r)**(n/2) # un-nornalised version of the coeficient definition from eq 7.67 in G+K
            psi_out[i_photon]=beamsplit_prefactor*C_squeeze # element of state vector proportional to product of two terms
            squeeze_norm_sq=squeeze_norm_sq+C_squeeze**2 # add square of coefficient
            n_new=n+1 # value of n to be used in 7.60 for next term in the recursion
            C_squeeze=-nu_over_mu*np.sqrt(n_new/(n_new+1))*C_squeeze # apply recursion relationship from 7.60
    norm_psi=np.sqrt(sum(abs(psi_out)**2))
    psi_out=psi_out/norm_psi # normalise output wavefunction
    if return_p:
        p_k=(norm_psi**2)*(np.sin(theta_subtract)**(2*k))/np.cosh(xi_abs) # multiply by additional k dependent factor related to probability of photons splitting
        return(psi_out,p_k)
    else:
        return psi_out
    
def squeeze_beam_split_analytic_range(n_sub_detect,n_hat_squeeze=np.infty,theta=0,theta_subtract=np.pi/5,n_sub=1001): # uses the analytic formula to calculate a list of output wavefunctions and probabilites (assuming no appriciable probabilities outside of the range
    '''
        n_sub_out is the maximum measured number of photons plus one
        n_hat_squeeze is the number of photons of squeezing added, used to back-calculate the amplitude of squeezing, can be sent to np.infty for infinite squeezing limit, this is the default behaviour
        theta is the squeeezing angle
        theta_subtract is the beamsplitting angle used when subtracting the photons
        n_sub is the truncated size of the final state vector (maximum number of photons plus one)
        return_norm is a flag to return the normalisation factor (useful in calculating probabilities)
    '''
    p_list=[None]*n_sub # array for probabilites
    psi_sub_renorm_list=[None]*n_sub # list for wavefunctions
    for detect_number in range(n_sub_detect): # loop through detection numbers
        (psi_out,p_k)=squeeze_beam_split_analytic(detect_number,n_hat_squeeze=n_hat_squeeze,theta=theta,theta_subtract=theta_subtract,n_sub=n_sub,return_p=True) # analytically calculated data
        psi_sub_renorm_list[detect_number]=copy.copy(psi_out) # add to list of renormalised states
        p_list[detect_number]=copy.copy(p_k) # add to list of probabilities
    return[psi_sub_renorm_list,p_list]
        

def prepare_shifted_kitten_state_photon_number(n_hat_disp,n_hat_kit,n_sub,phi_disp=0,phi_squeeze=0): # prepares a shifted Schrodinger kitten state with a given number of photons from squeezing and from displacement
    '''
        n_hat_disp is the (average) number of photons from displacement used to calculate the displacement magnitude
        n_hat_squeeze is the (average) number of photons from squeezing used to calculate the squeezing magnitude
        n_sub is the size of the subspace (maximum allowed photon number plus 1
        phi_disp is the phase of the displacement (defaults to zero)
        phi_squeeze is the phase of the squeezing (defaults to zero)
    '''
    (alpha_abs,xi_abs)=find_disp_and_squeezing_photon_number(n_hat_disp,n_hat_kit+1) # calcuate the magnitude of the squeezing and displacement
    psi=prepare_squeezed_state_photon_number(0,n_hat_kit+1,n_sub,phi_disp=0,phi_squeeze=phi_squeeze) # squeezed vaccuum for photon subtraction
    a=make_annihiltion_Op(n_sub-1) # annihilation operator
    psi=a@psi # apply annihilation operator
    psi=psi/np.sqrt(sum(abs(psi)**2)) # renormalise after photon subtraction
    psi=apply_displacement_Op(alpha_abs*np.exp(1j*phi_squeeze),psi)
    return psi


def Gaussian_beamsplitting_output(alpha_1,xi_1,n_sub_1,alpha_2,xi_2,n_sub_2,theta): # calculates output from applying a beamsplitter to two Gaussian modes
    '''
        alpha_1 is the displacement of the first mode given as a complex number
        x_1 is the squeezing of the first mode given as a complex number
        n_sub_1 is the size of the first subspace (maximum number of photons minus 1)
        alpha_2 is the displacement of the second mode given as a complex number
        x_2 is the squeezing of the second mode given as a complex number
        n_sub_2 is the size of the second subspace (maximum number of photons minus 1)
        theta is the rotation angle of the beamsplitter given as a real number
        returns the wavefunction of the combined system
    '''
    psi=np.zeros(n_sub_1*n_sub_2) # create vector of correct size
    psi[0]=1 # start in vacuum
    psi=apply_squeezing_Op(xi_1,psi,n_after=n_sub_2) # squeezing on first mode
    psi=apply_displacement_Op(alpha_1,psi,n_after=n_sub_2) # displacement on first mode
    psi=apply_squeezing_Op(xi_2,psi,n_before=n_sub_1) # squeezing on second mode
    psi=apply_displacement_Op(alpha_2,psi,n_before=n_sub_1) # displacement on second mode
    psi=apply_beamsplitting_Op(theta,psi,n_before=[1,n_sub_1],n_after=[n_sub_2,1]) # apply beamsplitting
    return psi # return wavefunction
    
def apply_asym_gadget(psi1,psi2,theta_split=np.pi/5,theta_recomb=np.pi/10,n_sub_int=None,phi_int=0,psi_int=None): # applies a gadget which implements controlled loss based on the relative amplitude of two modes
    '''
        psi1 is the wavefunction of the state in the first mode
        psi2 is the wavefunction in the second mode
        psi_int is the wavefunction in each of the interaction modes (defaults to vacuum)
        theta_split is the strength of the beamsplitting which picks off some light from each mode
        theta_recomb is the strength of the beamsplitting which interferes the light from the other mode
        n_sub_int is the size of the subspace for the intermediate modes, defaults to None which sets the value equal to the size of psi1
        phi_int is the phase applied to the picked off light, defaults to zero which minimises loss if one mode is full and the other is empty
        
    '''
    n_sub_1=len(psi1) # size of first subspace
    n_sub_2=len(psi2) # size of second subspace
    if type(n_sub_int)==type(None): # if not supplied
        n_sub_int=n_sub_1 # set equal to number in subspace 1
    psi_tot=np.kron(psi2,psi1) # tensor product of two input modes
    if type(psi_int)==type(None): # default to vacuum in interaction modes
        psi_int=np.zeros(n_sub_int)
        psi_int[0]=1 # start with vacuum on the modes used for interactions
    psi_tot=np.kron(psi_int,psi_tot) # tensor in first interaction mode
    psi_tot=np.kron(psi_int,psi_tot) # tensor in second interaction mode
    psi_tot=apply_beamsplitting_Op(theta_split,psi_tot,n_before=[1,n_sub_1*n_sub_2],n_after=[n_sub_2*n_sub_int**2,n_sub_int]) # splitting off part of the first mode into the first coupling mode
    psi_tot=apply_beamsplitting_Op(theta_split,psi_tot,n_before=[n_sub_1,n_sub_1*n_sub_2*n_sub_int],n_after=[n_sub_int**2,1]) # splitting off part of the second mode into the second coupling mode
    if not phi_int==0:
        psi_tot=apply_time_evolution_Op(phi_int,psi_tot,n_before=n_sub_1*n_sub_2,n_after=n_sub_int) # apply phase to first coupling mode
        psi_tot=apply_time_evolution_Op(phi_int,psi_tot,n_before=n_sub_1*n_sub_2*n_sub_int,n_after=1) # apply phase to second coupling mode
    psi_tot=apply_beamsplitting_Op(theta_recomb,psi_tot,n_before=[n_sub_1,n_sub_1*n_sub_2],n_after=[n_sub_int**2,n_sub_int]) # combining the first coupling mode with the second mode
    psi_tot=apply_beamsplitting_Op(theta_recomb,psi_tot,n_before=[1,n_sub_1*n_sub_2*n_sub_int],n_after=[n_sub_2*n_sub_int**2,1]) # combining the second coupling mode with the first mode
    psi_tot=apply_beamsplitting_Op(np.pi/4,psi_tot,n_before=[n_sub_1*n_sub_2,n_sub_1*n_sub_2*n_sub_int],n_after=[n_sub_int,1]) # 50/50 beamsplitter on the coupling modes to erase information on how symmetric the state is
    return psi_tot
    
def two_mode_single_beamsplitter(psi1,psi2,theta=np.pi/5): # applies a gadget which implements controlled loss based on the relative amplitude of two modes
    '''
        psi1 is the wavefunction of the state in the first mode
        psi2 is the wavefunction in the second mode
        theta_split is the strength of the beamsplitting
    '''
    n_sub_1=len(psi1) # size of first subspace
    n_sub_2=len(psi2) # size of second subspace
    psi_tot=np.kron(psi2,psi1) # tensor product of two input modes
    psi_tot=apply_beamsplitting_Op(theta,psi_tot,n_before=[1,n_sub_1],n_after=[n_sub_2,1]) # apply a simple beamspitting between the two modes
    return psi_tot
    
def make_asym_gadget_mats(n_sub_1,n_sub_2,theta_split=np.pi/5,theta_recomb=np.pi/10,n_sub_int=None,phi_int=0): # creates the matrices necessary to apply a gadget which implments controlled loss
    '''
        n_sub_1 is the size of the vector describing the wavefunction of the state in the first mode
        n_sub_2 is the size of the vector describing the wavefunction of the state in the second mode
        theta_split is the strength of the beamsplitting which picks off some light from each mode
        theta_recomb is the strength of the beamsplitting which interferes the light from the other mode
        n_sub_int is the size of the subspace for the intermediate modes, defaults to None which sets the value equal to the size of psi1
        phi_int is the phase applied to the picked off light, defaults to zero which minimises loss if one mode is full and the other is empty
        returns matrices to exponetiate and to apply projections
    '''
    if n_sub_int==None: # if not supplied
        n_sub_int=n_sub_1 # set equal to one
    psi_tot=n_sub_1*n_sub_2*(n_sub_int**2) # size of the wavefunction vector
    exp_mat_list=[] # empty list for matrices to be exponentiated
    exp_mat_list=exp_mat_list+[apply_beamsplitting_Op(theta_split,psi_tot,n_before=[1,n_sub_1*n_sub_2],n_after=[n_sub_2*n_sub_int**2,n_sub_int],gen_mat=True)] # splitting off part of the first mode into the first coupling mode
    exp_mat_list=exp_mat_list+[apply_beamsplitting_Op(theta_split,psi_tot,n_before=[n_sub_1,n_sub_1*n_sub_2*n_sub_int],n_after=[n_sub_int**2,1],gen_mat=True)] # splitting off part of the second mode into the second coupling mode
    if not phi_int==0:
        exp_mat_list=exp_mat_list+[apply_time_evolution_Op(phi_int,psi_tot,n_before=n_sub_1*n_sub_2,n_after=n_sub_int,gen_mat=True)] # apply phase to first coupling mode
        exp_mat_list=exp_mat_list+[apply_time_evolution_Op(phi_int,psi_tot,n_before=n_sub_1*n_sub_2*n_sub_int,n_after=1,gen_mat=True)] # apply phase to second coupling mode
    exp_mat_list=exp_mat_list+[apply_beamsplitting_Op(theta_recomb,psi_tot,n_before=[n_sub_1,n_sub_1*n_sub_2],n_after=[n_sub_int**2,n_sub_int],gen_mat=True)] # combining the first coupling mode with the second mode
    exp_mat_list=exp_mat_list+[apply_beamsplitting_Op(theta_recomb,psi_tot,n_before=[1,n_sub_1*n_sub_2*n_sub_int],n_after=[n_sub_2*n_sub_int**2,1],gen_mat=True)] # combining the second coupling mode with the first mode
    exp_mat_list=exp_mat_list+[apply_beamsplitting_Op(np.pi/4,psi_tot,n_before=[n_sub_1*n_sub_2,n_sub_1*n_sub_2*n_sub_int],n_after=[n_sub_int,1],gen_mat=True)] # 50/50 beamsplitter on the coupling modes to erase which way information
    proj_mats_list=[photon_number_subspace_probabilities(None,n_sub_int,n_before=n_sub_1*n_sub_2,n_after=n_sub_int,gen_mats=True),photon_number_subspace_probabilities(None,n_sub_int,n_before=n_sub_1*n_sub_2*n_sub_int,n_after=1,gen_mats=True)] # matrices for projection
    return (exp_mat_list,proj_mats_list)
    
    
def psi_asym_gadget_squeeze_kit(psi_params,gadg_params): # applies interference gadget to states prepared using prepare_squeezed_state_photon_number
    '''
        psi_params is a list of two dictionaries, one for each state containing the inputs for prepare_squeezed_state_photon_number
        gadg_params is a dictionary with the inputs for apply_asym_gadget asside from psi1 and psi2 which are calculated as part of the function call if gadg_params['n_sub_int']==1 then instead of performing the full gadget the code just applies a single beamsplitter between the two modes with an angle defined by theta_split
    '''
    psi1=prepare_squeezed_state_photon_number(**psi_params[0]) # prepare first input wavefunction
    psi2=prepare_squeezed_state_photon_number(**psi_params[1]) # prepare second input wavefunction
    if 'n_sub_int' in gadg_params and gadg_params['n_sub_int']==1: # if the size of the interaction subspace is set to 1 then just interact the modes directly
        psi_tot=two_mode_single_beamsplitter(psi1,psi2,theta=gadg_params['theta_split'])
        return (psi_tot,psi1,psi2)
    elif len(psi_params)>2: # if a third set of parameters is provided for the interaction mode
        psi_int=prepare_squeezed_state_photon_number(**psi_params[2]) # prepare second input wavefunction
        psi_tot=apply_asym_gadget(psi1,psi2,psi_int=psi_int,**gadg_params)
        return (psi_tot,psi1,psi2,psi_int)
    else:
        psi_tot=apply_asym_gadget(psi1,psi2,**gadg_params) # applies the gadget
        return (psi_tot,psi1,psi2)
    
def photon_loss_rates_gadget(n_hats_disp,n_hats_squeeze,phis_disp=[[0,0],[0,0]],phis_squeeze=[[0,0],[0,0]],start_photons=[0,0],n_points=10,gadg_params={},n_sub=31,vacuum_inputs=[False,False],add_func=None): # applies interference gadget for range of parameters and calculates loss rate
    '''
        n_hats_disp is a list-of-list of length 2 where each list is length 2 or 3, the first list is the initial number of displacement photons added to each mode while the second is the final, for example [[5,0],[0,5]] would interpolate between 5 photons of displacement on the first and none on the second and none on the first and 5 on the second, the third entry of the sublist if present is the displacement for the interaction modes, which is set to be vacuum otherwise
        n_hats_squeeze is a list-of-list of length 2 where each list is also of length 2, the first list is the initial number of photons added by squeezing (assuming the system starts in vacuum) to each mode the second is the same but at the end, for example [[1,0],[0,1]] will interpolate between one photon of squeezing on the first state and none on the second to none on the first and one on the second, the third entry applies to the intraction modes as above
        phis_disp is the angle applied to the displacement given as a length-two list-of-lists as with the first two parameters, it controls the angle of the displacement, it defaults to all zeros, the third entry applies to the intraction modes as above
        phis_squeeze is a similar length 2 list-of-lists which gives the squeezing angles, the third entry applies to the intraction modes as above
        start_photons is a list of length 2 or 3 giving the number of photons to start with in each mode, defaults to [0,0], starting in vacuum in each mode, if a third value is given it applies to the interaction modes
        n_points is the number of points to use for the interpolation, defaults to 10
        gadg_params is the parameters to give the gadget, defaults to an empty dictionary, which means default values will be used for everything
        returns the number of photons lost and a dictionary containing other data
        vacuum inputs overrides the starting state for one (or both) of the input channels to allow the effect of interference to be evaluated
        add_func is an optional additional function which can be used to calculate other quantities from the full wavefunction
    '''
    loss_arr=np.zeros(n_points) # empty array for loss fractions
    photon_in_1_arr=np.zeros(n_points) # empty array for photons input in the first mode
    photon_in_2_arr=np.zeros(n_points) # empty array for photons input in the second mode
    photon_in_int_arr=np.zeros(n_points) # empty array for photons input in the interaction modes
    if not type(add_func)==type(None): # if an additional function has been provided
        add_data_list=[None]*n_points # empty list for additional data which the function calcualtes
    n_hat_disp_vals=[np.linspace(n_hats_disp[0][0],n_hats_disp[1][0],n_points),np.linspace(n_hats_disp[0][1],n_hats_disp[1][1],n_points)] # values for n_hat_disp for each input mode
    if len(n_hats_disp[0])>2: # if values have been provided for displacement
        n_hat_disp_vals=n_hat_disp_vals+[np.linspace(n_hats_disp[0][2],n_hats_disp[1][2],n_points)] # add parameters for interaction modes
    n_hat_squeeze_vals=[np.linspace(n_hats_squeeze[0][0],n_hats_squeeze[1][0],n_points),np.linspace(n_hats_squeeze[0][1],n_hats_squeeze[1][1],n_points)] # values for n_hat_disp for each input mode
    if len(n_hats_squeeze[0])>2: # if values have been provided for displacement
        n_hat_squeeze_vals=n_hat_squeeze_vals+[np.linspace(n_hats_squeeze[0][2],n_hats_squeeze[1][2],n_points)] # add parameters for interaction modes
    phi_disp_vals=[np.linspace(phis_disp[0][0],phis_disp[1][0],n_points),np.linspace(phis_disp[0][1],phis_disp[1][1],n_points)] # values for phi_disp for each input mode
    if len(phis_disp[0])>2: # if values have been provided for displacement
            phi_disp_vals=phi_disp_vals+[np.linspace(phis_disp[0][2],phis_disp[1][2],n_points)] # add parameters for interaction modes
    phi_squeeze_vals=[np.linspace(phis_squeeze[0][0],phis_squeeze[1][0],n_points),np.linspace(phis_squeeze[0][1],phis_squeeze[1][1],n_points)] # values for phi_squeeze for each input mode
    if len(phis_squeeze[0])>2: # if values have been provided for displacement
        phi_squeeze_vals=phi_squeeze_vals+[np.linspace(phis_squeeze[0][2],phis_squeeze[1][2],n_points)] # add parameters for interaction modes
    for i_point in range(n_points): # loop through all points
        if len(start_photons)>2 or len(n_hat_disp_vals)>2 or len(n_hat_squeeze_vals)>2 or len(phi_disp_vals)>2 or len(phi_squeeze_vals)>2: # if any information has been provided about the interaction mode
            psi_params=[{},{},{}] # list of empty dictionaries
            if 'n_sub_int' in gadg_params and (not type(gadg_params['n_sub_int'])==type(None)): # if a value has been supplied
                psi_params[2]['n_sub']=gadg_params['n_sub_int'] # use supplied value
            else:
                psi_params[2]['n_sub']=n_sub # otherwise default to the same number as the other two modes
        else:
            psi_params=[{},{}] # list of empty dictionaries
        psi_params[0]['n_sub']=n_sub
        psi_params[1]['n_sub']=n_sub
        psi_params[0]['vacuum']=vacuum_inputs[0]
        psi_params[1]['vacuum']=vacuum_inputs[1]
        for i_param in range(len(start_photons)):
            psi_params[i_param]['start_photons']=start_photons[i_param] # start photons in each mode
        for i_param in range(len(n_hat_disp_vals)):
            psi_params[i_param]['n_hat_disp']=n_hat_disp_vals[i_param][i_point] # displacement photons in each mode
        for i_param in range(len(n_hat_squeeze_vals)):
            psi_params[i_param]['n_hat_squeeze']=n_hat_squeeze_vals[i_param][i_point] # squeezing photons in each mode
        for i_param in range(len(phi_disp_vals)):
            psi_params[i_param]['phi_disp']=phi_disp_vals[i_param][i_point] # displacement angle for each mode
        for i_param in range(len(phi_squeeze_vals)):
            psi_params[i_param]['phi_squeeze']=phi_squeeze_vals[i_param][i_point] # squeezing angle for each mode
        if len(psi_params)>2: # if a third set of parameters is provided for the interaction mode
            (psi_tot,psi1,psi2,psi_int)=psi_asym_gadget_squeeze_kit(psi_params,gadg_params) # calculate wavefunctions
            n_sub_int=len(psi_int) # size of interaction subspace
            photon_in_int_arr[i_point]=2*mean_photon_count_subspace(psi_int,n_sub_int) # number of photons coming in from interaction subspaces
        else:
            (psi_tot,psi1,psi2)=psi_asym_gadget_squeeze_kit(psi_params,gadg_params) # calculate wavefunctions
            photon_in_int_arr[i_point]=0 # no photons in interaction mode
        n_sub_1=len(psi1) # size of first subspace
        photon_in_1_arr[i_point]=mean_photon_count_subspace(psi1,n_sub_1) # number of photons coming in from first subspace
        n_sub_2=len(psi2) # size of second subspace
        photon_in_2_arr[i_point]=mean_photon_count_subspace(psi2,n_sub_2) # number of photons coming in from second subspace
        n_sub_int=int(round(np.sqrt(len(psi_tot)/(n_sub_1*n_sub_2)))) # size of each interaction subspace
        if 'n_sub_int' in gadg_params and gadg_params['n_sub_int']==1: # if the interaction modes are not present that the loss fraction is the number of photons in the second mode
            n_photon_loss=mean_photon_count_subspace(psi_tot,n_sub,n_before=n_sub_1,n_after=1) # mean number of photons in second mode
            loss_arr[i_point]=n_photon_loss
        else:
            n_photon_loss=mean_photon_count_subspace(psi_tot,n_sub,n_before=n_sub_1*n_sub_2,n_after=n_sub_int)+mean_photon_count_subspace(psi_tot,n_sub,n_before=n_sub_1*n_sub_2*n_sub_int,n_after=1) # total number of lost photons
            loss_arr[i_point]=n_photon_loss # number of photons lost
        if not type(add_func)==type(None): # if an additional function has been provided
            if 'n_sub_int' in gadg_params and gadg_params['n_sub_int']==1: # if the interaction modes are not present that the loss fraction is the number of photons in the second mode
                add_data_list[i_point]=add_func(psi_tot,n_sub_1,n_sub_2) # apply function and store result in list
            else:
                add_data_list[i_point]=add_func(psi_tot,n_sub_1,n_sub_2,n_sub_int) # apply function and store result in list
    data_dict={} # dictionary for additional data
    data_dict['photon_in_1_arr']=photon_in_1_arr
    data_dict['photon_in_2_arr']=photon_in_2_arr
    data_dict['photon_in_int_arr']=photon_in_int_arr
    data_dict['n_hat_disp_vals']=n_hat_disp_vals
    data_dict['n_hat_squeeze_vals']=n_hat_squeeze_vals
    data_dict['phi_disp_vals']=phi_disp_vals
    data_dict['phi_squeeze_vals']=phi_squeeze_vals
    if not type(add_func)==type(None): # if an additional function has been provided
        data_dict['add_data_list']=add_data_list # pass additional data as part of data_dict
    return (loss_arr,data_dict)
    
def photon_loss_rates_int_gadget(n_hats_disp,n_hats_squeeze,phis_squeeze=[[0,0],[0,0]],start_photons=[0,0],n_points=10,gadg_params={},n_sub=31,add_func=None): # calculates the number of photons lost due to inteference by taking the difference of the interference gadget result and the same result with a phase shift of pi/2 on one displacement to remove interference
    '''
        n_hats_disp is a list-of-list of length 2 where each list is length 2 or 3, the first list is the initial number of displacement photons added to each mode while the second is the final, for example [[5,0],[0,5]] would interpolate between 5 photons of displacement on the first and none on the second and none on the first and 5 on the second, the third entry of the sublist if present is the displacement for the interaction modes, which is set to be vacuum otherwise
        n_hats_squeeze is a list-of-list of length 2 where each list is also of length 2, the first list is the initial number of photons added by squeezing (assuming the system starts in vacuum) to each mode the second is the same but at the end, for example [[1,0],[0,1]] will interpolate between one photon of squeezing on the first state and none on the second to none on the first and one on the second, the third entry applies to the intraction modes as above
        phis_squeeze is a similar length 2 list-of-lists which gives the squeezing angles, the third entry applies to the intraction modes as above
        start_photons is a list of length 2 or 3 giving the number of photons to start with in each mode, defaults to [0,0], starting in vacuum in each mode, if a third value is given it applies to the interaction modes
        n_points is the number of points to use for the interpolation, defaults to 10
        gadg_params is the parameters to give the gadget, defaults to an empty dictionary, which means default values will be used for everything
        returns the number of photons lost and a dictionary containing other data
        add_func is an optional additional function which can be used to calculate other quantities from the full wavefunction
    '''
    (loss_arr_int,data_dict_int)=photon_loss_rates_gadget(n_hats_disp,n_hats_squeeze,phis_disp=[[0,0],[0,0]],phis_squeeze=phis_squeeze,start_photons=start_photons,n_points=n_points,gadg_params=gadg_params,n_sub=n_sub,add_func=add_func) # data for setting with intereference
    (loss_arr_vac0,data_dict_vac0)=photon_loss_rates_gadget(n_hats_disp,n_hats_squeeze,phis_disp=[[0,np.pi/2],[0,np.pi/2]],phis_squeeze=phis_squeeze,start_photons=start_photons,n_points=n_points,gadg_params=gadg_params,n_sub=n_sub,vacuum_inputs=[True,False],add_func=add_func) # data with vacuum on one input
    (loss_arr_vac1,data_dict_vac1)=photon_loss_rates_gadget(n_hats_disp,n_hats_squeeze,phis_disp=[[0,np.pi/2],[0,np.pi/2]],phis_squeeze=phis_squeeze,start_photons=start_photons,n_points=n_points,gadg_params=gadg_params,n_sub=n_sub,vacuum_inputs=[False,True],add_func=add_func) # data for vacuum on other input
    loss_diff_arr=loss_arr_int-loss_arr_vac0-loss_arr_vac1 # difference between the loss rates
    return (loss_diff_arr,data_dict_int,[data_dict_vac0,data_dict_vac1]) # return difference in loss rate
    
def prepare_two_mode_photon_subtract_states(n_hat_squeeze=5,theta_subtract=np.pi/5,displacement_photons_add=5,squeezing_photons_add=0,squeeezing_photon_angle=0,n_sub=31, add_match=True, const_disp=True,return_kit=False, analytic_def=True): # prepares a list of two-mode states prepared from performing photon subtraction on a squeezed state and then
    '''
        n_hat_squeeze is the number of photons of initial squeezing
        theta_subtract is the angle used in the beamsplitter which subtracts the photons 
        displacement_photons_add is the number of photons of displacement to add
        squeezing_photons_add is the number of photons of squeezing to perform on the added state
        returns list of output wavefunctions for given photon number measurements and list of probabilities of each
        add_match is a flag to add a matching number of photons equal to the average number in the kitten like state (makes comparisons more fair between different numbers of subtracted photons)
        const_disp adds a constant displacement corresponding to the number of displacement photons which would be added (more photons when displacement is higher)
        return_kit is a flag which determines whether or not to return the kitten-like states before beamsplitting
        analytic_def is a flag to use an analytic definition, which tends to be more stable against truncation error and can support higher photon number 
    '''
    if analytic_def:
            [psi_sub_renorm_list,p_list]=squeeze_beam_split_analytic_range(n_sub_detect=n_sub,n_hat_squeeze=n_hat_squeeze,theta=0,theta_subtract=theta_subtract,n_sub=n_sub) # wavefunctions and probabilities from analytic definition
    else: # otherwise prepare using numerical matrix multiplication (less stable against truncation errors)
        psi_params=[{'n_hat_squeeze':n_hat_squeeze,'n_sub':n_sub},{'n_sub':n_sub}] # parameters for initial state, left empty because vacuum in second mode
        gadg_params={'theta_split': theta_subtract,'n_sub_int':1} # splitting angle setting n_sub_int to 1 removes interaction modes
        (psi_init,psi1,psi2)=psi_asym_gadget_squeeze_kit(psi_params,gadg_params) # create initial modes by applying a beamsplitter between two start states
        [psi_sub_renorm_list,p_list]=state_vector_sub_2_photon_measure(psi_init,n_sub,n_sub) # calculate the probabilities and renormalised wavefunctions for different photon number measurements
    if not add_match: # if not adding a matching number of photons
        psi_coh=prepare_squeezed_state_photon_number(n_hat_disp=displacement_photons_add,n_hat_squeeze=squeezing_photons_add,phi_squeeze=squeeezing_photon_angle,phi_disp=0,n_sub=n_sub) # prepare an additional coherent state for the beamsplitter to convert to spatial modes
    psi_final_list=[]
    for psi_in in psi_sub_renorm_list:
        if add_match: # if we are adding a matching number of photons
            n_photon_in=mean_photon_count_subspace(psi_in,len(psi_in))
            if const_disp: # adding the same displacement to each rather than the same photon number
                n_hat_disp=(np.sqrt(displacement_photons_add)+np.sqrt(n_photon_in))**2 # add ampltudes and calculate photon number
            else:
                n_hat_disp=displacement_photons_add+n_photon_in
            psi_coh=prepare_squeezed_state_photon_number(n_hat_disp=n_hat_disp,n_hat_squeeze=squeezing_photons_add,phi_squeeze=squeeezing_photon_angle,phi_disp=0,n_sub=n_sub) # prepare an additional coherent state for the beamsplitter to convert to spatial modes
        psi_comb=np.kron(psi_coh,psi_in) # tensor wih coherent state
        psi_final=apply_beamsplitting_Op(np.pi/4,psi_comb,n_before=[1,n_sub],n_after=[n_sub,1]) # perform beamsplitting
        psi_final=apply_time_evolution_Op(np.pi/2,psi_final,n_after=n_sub)
        psi_final_list=psi_final_list+[psi_final] # add to list
    if return_kit:
        return(psi_final_list,psi_sub_renorm_list,p_list)
    else:
        return(psi_final_list,p_list)
        
def gadget_photon_number_mat(psi_tot,n_sub_1,n_sub_2,n_sub_int): # produces a correlation matrix for the number of photons in each mode, with interaction modes traced out
    '''
        psi_tot is the total wavefunction
        n_sub_1 is the size of the first channel (number of allowed photons+1)
        n_sub_2 is the size of the second channel (number of allowed photons+1)
        n_sub_int is the size of each interaction channel (number of allowed photons+1)
        returns traced out probabilities for number of photons in each output mode
    '''
    probs=abs(psi_tot)**2 # probability corresponding to each entry
    p=photon_number_prob_sum(n_sub_int**2,n_before=n_sub_1*n_sub_2,n_after=1) # matrix which performs summation over both interaction channels (equivalent to tracing out but only calculates diagonal density matrix elements)
    probs=p@probs # multiply to project out
    probs_mat=np.reshape(probs,(n_sub_1,n_sub_2)) # reshape into a matrix
    return probs_mat
    
def total_output_photon_probs(psi_tot,n_sub_1,n_sub_2,n_sub_int): # calculate the probablity of having a different total number of photons leaving from the interaction modes
    '''
        psi_tot is the total wavefunction
        n_sub_1 is the size of the first channel (number of allowed photons+1)
        n_sub_2 is the size of the second channel (number of allowed photons+1)
        n_sub_int is the size of each interaction channel (number of allowed photons+1)
    '''
    probs=abs(psi_tot)**2 # array of probabilities
    n_photon_max=2*n_sub_int-2 # maximum number of photons leaving through interaction modes
    n_photon_int_1_arr=np.kron(np.ones(n_sub_int),np.kron(np.array(range(n_sub_int)),np.ones(n_sub_1*n_sub_2))) # number of photons in interaction mode 1 for each entry
    n_photon_int_2_arr=np.kron(np.array(range(n_sub_int)),np.ones(n_sub_int*n_sub_1*n_sub_2)) # number of photons in interaction mode 1 for each entry
    n_photon_int_total_arr=n_photon_int_1_arr+n_photon_int_2_arr # total number of photons in the interaction modes for each entry
    p_list=[] # empty list for probabilities
    for i_photon_n in range(0,n_photon_max+1): # loop through total number of photons in output mode
        p_list=p_list+[sum(probs[n_photon_int_total_arr==i_photon_n])] # add to list of probabilities
    return p_list

def state_vector_balance_photon_measure(psi_tot,n_sub_1,n_sub_2,n_sub_int): # state vector of encoded subspace with balanced measurement values
    '''
        psi_tot is the total wavefunction
        n_sub_1 is the size of the first channel (number of allowed photons+1)
        n_sub_2 is the size of the second channel (number of allowed photons+1)
        n_sub_int is the size of each interaction channel (number of allowed photons+1)
    '''
    n_photon_max=2*n_sub_int-2 # maximum number of photons leaving through interaction modes
    psi_renorm_list=[] # empty list for renormalised psi
    p_list=total_output_photon_probs(psi_tot,n_sub_1,n_sub_2,n_sub_int)
    for i_photon_n in range(0,n_photon_max+1): # loop through total number of photons in output mode
        n_photon_int_1=np.floor(i_photon_n/2) # first interaction channel
        p1=photon_number_subspace_project(n_photon_int_1,n_sub_int,n_before=n_sub_1*n_sub_2,n_after=n_sub_int) # project out first interaction subspace
        n_photon_int_2=np.ceil(i_photon_n/2) # second interaction channel
        p2=photon_number_subspace_project(n_photon_int_2,n_sub_int,n_before=n_sub_1*n_sub_2,n_after=1) # project out second interaction subspace
        psi_proj=p2@p1@psi_tot # projected out version
        psi_renorm_list=psi_renorm_list+[psi_proj/np.sqrt(sum(abs(psi_proj)**2))] # add to list of renormalised wavefunctions
    return [psi_renorm_list,p_list]
    
def total_output_photon_probs_2(psi_tot,n_sub_1,n_sub_2): # calculate the probablity of having a different total number of photons leaving from the interaction modes
    '''
        psi_tot is the total wavefunction
        n_sub_1 is the size of the first channel (number of allowed photons+1)
        n_sub_2 is the size of the second channel (number of allowed photons+1)
    '''
    probs=abs(psi_tot)**2 # array of probabilities
    n_photon_max=n_sub_2 # maximum number of photons leaving through interaction modes
    n_photon_2_arr=np.kron(np.array(range(n_sub_2)),np.ones(n_sub_1)) # number of photons in mode 2
    p_list=[] # empty list for probabilities
    for i_photon_n in range(0,n_photon_max+1): # loop through total number of photons in output mode
        p_list=p_list+[sum(probs[n_photon_2_arr==i_photon_n])] # add to list of probabilities
    return p_list
    
def state_vector_sub_2_photon_measure(psi_tot,n_sub_1,n_sub_2): # state vector of encoded subspace with balanced measurement values
    '''
        psi_tot is the total wavefunction
        n_sub_1 is the size of the first channel (number of allowed photons+1)
        n_sub_2 is the size of the second channel (number of allowed photons+1)
    '''
    n_photon_max=n_sub_2-1 # maximum number of photons leaving through second mode
    psi_renorm_list=[] # empty list for renormalised psi
    p_list=total_output_photon_probs_2(psi_tot,n_sub_1,n_sub_2)
    for i_photon_n in range(0,n_photon_max+1): # loop through total number of photons in output mode
        p1=photon_number_subspace_project(i_photon_n,n_sub_2,n_before=n_sub_1,n_after=1) # project out second subspace
        psi_proj=p1@psi_tot # projected out version
        psi_renorm_list=psi_renorm_list+[psi_proj/np.sqrt(sum(abs(psi_proj)**2))] # add to list of renormalised wavefunctions
    return [psi_renorm_list,p_list]
    
def calculate_squeezing_displacement_diff(disp_photons_init,disp_photons_target): # determines the amount of displacement (in photons) needed to go from an initial to a final displacement
    '''
        disp_photons_init is the number of photons in the initial state
        disp_photons_final is the number of photons in the final state
        assumes squeezing in the same direction
    '''
    disp_ratio=disp_photons_target/disp_photons_init # ratio of target number of displacement photons to initial ones
    xi=np.log(disp_ratio)/2 # take the log of the ratio to find the anount of squeezing which is needed, divide by 2 because photon number is the square root of the displacment
    return xi # return squeezing (up to overall phase)
    
def calculate_fraction_from_squeezing_d_r(d_0,r_0,r): # fraction of photons from squeezing with initial displacement d_0 and anti-squeezing r_0, after further squeezing of r
    '''
        d_0 is the initial displacement
        r_0 is the initial anti-squeezing
        r is the additional anti-suqeezing applied
        returns the fraction of total photons from anti-squeezing
    '''
    if np.isfinite(r): # for finite amount of anti-squeezing
        squeeze_frac=(np.sinh(r+r_0))**2/((d_0**2)*np.exp(2*r)+(np.sinh(r+r_0))**2)
    else: # for an infinite amount of squeezing
        squeeze_frac=np.exp(2*r_0)/(4*(d_0**2)+np.exp(2*r_0))
    return squeeze_frac
    
def calculate_fraction_from_squeezing_n_tot_frac(n_tot,frac_asq_init,r): # wrapper to calculate the fraction from squeezing based on total initial photon number and initial fraction from anti-squeezing
    '''
        n_tot is the total number of initial photons from both displacement and squeezing
        frac_asq_init is the initial fraction from anti-squeezing (should take a value between 0 and 1 inclusive)
        r is the amount of additional anti-squeezing to do
    '''
    d_0=np.sqrt(n_tot*(1-frac_asq_init)) # the initial displacement is the square root of the total number of photons of displacement
    r_0=np.arcsinh(np.sqrt(n_tot*frac_asq_init)) # the initial squeezing is the arcsinh of the square root of the number of photons of squeezing
    squeeze_frac=calculate_fraction_from_squeezing_d_r(d_0,r_0,r) # calculate fraction
    return squeeze_frac
    


    
    
