import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import scipy.linalg as linalg_d
import time
import scipy.optimize as opt
import scipy.special as special
import copy
import QO_trunc_functions as QOtrunc # base module for truncated quantum optics


def pure_state_to_vectorised_rho(psi): # converts a pure state to a vectorised density matrix using tensor products
    '''
        takes state vector psi as an input and converts it to a vectorised representation of a density matrix created from the outer product
    '''
    rho_v=np.kron(psi,psi.conj()) # tensor product with it's complex conjugate
    return rho_v 

def vectorized_rho_to_matrix_rho(rho_v): # reshapes a vectorised version of a density matrix into a density matrix (uses a dense representation)
    '''
        rho_v is a vectorized version of a densiity matrix
    '''
    lin_size=int(np.sqrt(len(rho_v))) # size of one side of the density matrix
    rho_m=rho_v.reshape([lin_size,lin_size])
    return rho_m
    
def matrix_rho_to_vector_rho(rho_m): # reshapes a matrix version into a vector version
    '''
        rho_m is the density matrix expressed as an array
    '''
    lin_size=rho_m.shape[0] # size of one side of the matrix
    rho_v=rho_m.reshape([lin_size**2,])
    return rho_v
    
def convert_rho_to_vector_idem(rho): # additional logic around function to make rho into vector form to make it idempotent
    '''
        rho is a density matrix in array or vector format
        returns the (possibly unchanged) density matrix in array format
    '''
    if not isinstance(rho,int) and len(rho.shape)==2: # if given as a matrix, special case for integer so we won't try to calculate shape of an integer
        rho_v=matrix_rho_to_vector_rho(rho) # convert to vector
    else:
        rho_v=rho # take as-is
    return rho_v
    
def convert_rho_to_matrix_idem(rho): # additional logic around function to make rho into matrix form to make it idempotent
    '''
        rho is a density matrix in array or vector format
        returns the (possibly unchanged) density matrix in array format
    '''
    if not isinstance(rho,int) and not len(rho.shape)==2: # if not given as a matrix
        rho_m=vectorized_rho_to_matrix_rho(rho) # convert to matrix
    else:
        rho_m=rho # take as-is
    return rho_m
    
def get_subspace_photons_rho(rho,n_before=1,n_after=1): # calculates the maximum number of photons in a vectorized density matrix
    '''
        rho is a density matrix in either matrix or vector form
        n_after is the number of elements occuring after in the tensor product
        n_before is the number of elements occuring before in the tensor product
    '''
    rho_v=convert_rho_to_vector_idem(rho) # convert to vector form if not already
    tensor_product_size=QOtrunc.get_subspace_photons(rho_v,n_before=n_before,n_after=n_after)+1 # take advantage of existing function in QOtrunc, but shift by one
    matrix_size=np.sqrt(tensor_product_size) # since it is vectorized the density matrix size is the square root of the size used in the tensor product
    if abs(matrix_size-round(matrix_size))>(10**-10): # if the size is not an integer to within numerical rounding error
        raise RuntimeError("vectorized terms could not have arisen from a square density matrix")
    else:
        return int(round(matrix_size-1)) # round and convert to integer
    
def build_supOP(leftOp,rightOp): # builds a superoperator out of right and left operations
    '''
        leftOp is the operator acting on the left subspace as a sparse matrix, can be set to None if the operator acts only to the right
        rightOp is the operator acting on the right subspace as a sparse matrix, can be set to None if the operator acts only to the left
    '''
    if type(leftOp)==type(None): # if right subspace only
        leftOp=sparse.eye(rightOp.shape[0],format='csr') # identity operation
    if type(rightOp)==type(None): # if left subspace only
        rightOp=sparse.eye(leftOp.shape[0],format='csr') # identity operation
    supOp=sparse.kron(leftOp,rightOp.T,format='csr') # note that right subspace gets transposed
    return supOp

def build_lindblad_supOp(L): # takes a lindblad operator L and builds a superoperator in Lindblad form
    '''
        L is a sparse matrix for the operator used to generate the Lindbladian
    '''
    L_d=L.conj().T  # define L dagger once
    prod_Op=L_d@L # operator built from a product of terms
    lind_supOp=build_supOP(L,L_d)-0.5*(build_supOP(prod_Op,None)+build_supOP(None,prod_Op)) # definition of the Lindbladian
    return lind_supOp

def build_coherent_supOp(H): # takes a Hamiltonian H and builds the coherent part of a Lindblad master equation (effectively a von Neumann equation)
    '''
        H is a Hamiltonian, assumed to be Hermitian
    '''
    coh_supOp=-1j*(build_supOP(H,None)-build_supOP(None,H)) # commutator term with H
    return coh_supOp

def supOP_time_evolution(rho_v,supOp,t=1): # performs time evolution by exponentiating a superoperator over a time t
    '''
        rho_v is a vectorised density matrix
        supOp is a superoperator expressed as a matrix
        t is the time to evolve for (defaults to 1)
    '''
    rho_v_f=linalg.expm_multiply(t*supOp,rho_v) # exponentiate the superoperator
    return rho_v_f
    
def psi_gen_to_supOp(gen_mat): # takes the generator matrix for coherent evolution of a state vector and converts it to the relevant superoperator
    '''
        gen_mat is an anti-Hermitian generator matrix, assumed to be in a sparse format
    '''
    U=linalg_d.expm(gen_mat.todense()) # perform matrix exponentiation with dense matrix routine
    U_d=U.conj().T # U dagger
    U=sparse.csr_array(U) # convert to sparse format
    U_d=sparse.csr_array(U_d) # convert to sparse format
    supOp_U_gen=build_supOP(U,U_d) # construct the superoperator to perform the evolution
    return supOp_U_gen
    
    
def calculate_von_Neumann_entropy(rho): # calculates the von Neumann entropy automatically detects if rho is expressed as a matrix of a vector, calcualted in base 2 so that it has a natural interpretation in terms of bits
    '''
        rho is the density matrix expressed either as a dense matrix or a vector
        outputs base 2 von Neumann entropy
    '''
    rho_m=convert_rho_to_matrix_idem(rho) # idempotent conversion fuction
    [eig_vals,eig_vecs]=linalg_d.eigh(rho_m) # note that the use of eigh implicitly implies the matrix is hermitian
    log_vals=np.log2(eig_vals) # values of base 2 logarithm
    log_vals[np.isinf(log_vals)]=1 # regularise infinite values to prevent nans
    log_vals[np.isnan(log_vals)]=1 # regularise infinite values to prevent nans
    s=-eig_vals@log_vals # base 2 von Neumann entropy
    return s
    
def calculate_Op_expectation(rho,Op): # calculates the expectation value of an operator with respect to rho
    '''
        rho is the density matrix expressed either as a dense matrix or a vector
        Op is a Hermitian operator
        outputs the expecation value of Op
    '''
    rho_m=convert_rho_to_matrix_idem(rho) # idempotent conversion fuction
    expt=sum(np.diag(rho_m@Op)) # trace of the product of the density matrix with the operator
    return expt
    
def calculate_g_2(rho): # calculates the second order coherence function (\tau=0 because single mode) 5.93 (and 7.105) of G+K
    ''' 
        rho is the density matrix expressed either as a dense matrix or a vector 
        returns the numerical value of the second order coherence function
    '''
    rho_m=convert_rho_to_matrix_idem(rho) # idempotent conversion fuction
    n_trunc=rho_m.shape[0]-1 # maximum number of photons in Hilbert space
    a=QOtrunc.make_annihiltion_Op(n_trunc=n_trunc) # annihilation operator
    a_d=a.conj().T # creation operators
    n=calculate_Op_expectation(rho,a_d@a) # photon number expectation
    g_2=calculate_Op_expectation(rho,a_d@a_d@a@a)/(n**2) # 5.93 (and 7.105) of G+K
    return g_2
    
def calculate_displacement_and_covariance_single_mode_mixed(rho): # calculates the displacement and covariance matrix of a single mode mixed state
    '''
        rho is the density matrix in either matrix or vector form
    '''
    rho_m=convert_rho_to_matrix_idem(rho) # idempotent conversion fuction
    n_sub=rho_m.shape[0]
    Quad_list=[None]*2 # empty list for quadrature operators
    (Quad_list[0],Quad_list[1])=QOtrunc.make_X1_X2_ops(n_sub) # quadrature operators
    d_arr=np.zeros(2) # array for dispalcements
    for i_d in range(2): # loop through quadrature operators
        d_arr[i_d]=np.real(calculate_Op_expectation(rho_m,Quad_list[i_d])) # calcualte quadratures
    sigma_arr=np.zeros([2,2])
    for i_cov in range(2): # first index for finding covanriances
        for j_cov in range(i_cov,2): # second index for finding covanriances
            sigma_arr[i_cov,j_cov]=2*np.real((calculate_Op_expectation(rho,Quad_list[i_cov]@Quad_list[j_cov])+calculate_Op_expectation(rho,Quad_list[j_cov]@Quad_list[i_cov]))/2-d_arr[i_cov]*d_arr[j_cov]) # note factor of sqrt(2) difference in p and q from G+K X1 and X2 definition
            sigma_arr[j_cov,i_cov]=sigma_arr[i_cov,j_cov] # symmetric by definition
    return (d_arr,sigma_arr)
    
def remove_squeezing_cov(sigma_arr): # undoes the squeezing on covariance matrix using equation 50 arXiv:2102.05748
    '''
        sigma_arr is the covariance matrix
    '''
    (xi_abs,theta)=QOtrunc.find_squeezing_single_mode(sigma_arr) # calculate the squeezing parameters
    theta_rev=theta+np.pi # ssqueezing to remove original squeezing
    F=np.cosh(xi_abs)*np.eye(2)-np.sinh(xi_abs)*np.array([[np.cos(theta_rev),np.sin(theta_rev)],[np.sin(theta_rev),-np.cos(theta_rev)]]) # matrix to undo the squeezing based on equation 50 of arXiv:2102.05748
    sigma_rev=F@sigma_arr@F.T # apply operation (equation 22 of arXiv:2102.05748)
    return sigma_rev
    
def extract_Gaussian_state_parameters(sigma_arr): # extracts the parameters (assuming a Gaussian state) from a covariance matrx
    '''
        sigma_arr is the covariance matrix
    '''
    (xi_abs,theta)=QOtrunc.find_squeezing_single_mode(sigma_arr) # squeezing magnitude and angle found the same way as in the pure state case
    id_component=(sigma_arr[0,0]+sigma_arr[1,1])/2 # calculate the mean of the diagonal elements to estimate the temperature for a thermal component
    sigma_rev=remove_squeezing_cov(sigma_arr) # remove squeezing to estimate number of photons from thermal effects
    n_bar_therm=(sigma_rev[0][0]+sigma_rev[1][1])/2-1/2 # average number of thermal photons based on identity component of covariance
    if n_bar_therm<10**-9: # if the value is too small don't apply formula as it will lead to errors
        T_therm=0 # temperature of 0
    else:
        T_therm=-1/np.log(n_bar_therm/(1+n_bar_therm)) # from rearranging equation 2.143 of G+K
    return (xi_abs,theta,T_therm)
    
def create_thermal_rho(T,n_trunc,diagonal_only=False): # creats a thermal state density matrix in a subspace which is truncated at a maximum of ntrunc photons
    '''
        T is the temperature
        ntrunc is the maximum number of photons
        returns the density matrix as a square array
        diagonal_only is a flag which causes this function to return only the diagnal rather than the matrix if set to true
    '''
    if T==0: # if exactly zero return vacuum
        if diagonal_only:
            P_n=np.zeros(n_trunc+1)
            P_n[0]=1
            return P_n
        else:
            rho_m=np.zeros([n_trunc+1,n_trunc+1])
            rho_m[0,0]=1
            return rho_m
    Z=1/(1-np.exp(-1/T)) # partion function from equation 2.136 in G+K (factor of np.exp(-1/(2*T)) removed for numerical stability)
    P_n=[np.exp(-(n)/T)/Z for n in range(n_trunc+1)] # P_n from equation 2.137 in G+K, (additive factor of 1/2 removed for numerical stability)
    if diagonal_only: # if asked for only the diagonal
        return P_n
    rho_m=np.diag(P_n)
    return rho_m
    
def make_squeezed_vac_psi_analytic(xi_abs,theta_squeeze,n_trunc): # creates a suqeezed state analytically (more numerically stable than applying suqeezing to vacuum
    '''
        xi_abs is the absolute magnitude of the squeezing
        theta_squeeze is the angle of the squeezing
        n_trunc is the maximum number of photons to allow
    '''
    psi_out=np.zeros(n_trunc+1,dtype=complex) # initialize state as all zeros vector
    C_squeeze=1 # initialise the coefficiants for recursive calculation
    nu_over_mu=np.exp(1j*theta_squeeze)*np.tanh(xi_abs) # definition of the ratio of these coefficients from G+K
    squeeze_norm_sq=0 # squared norm for squeezed state (used in calculating probability)
    for i_photon in range(n_trunc+1):
        if i_photon%2==0: # if i_photon is even
            psi_out[i_photon]=copy.copy(C_squeeze)
            #squeeze_norm_sq=squeeze_norm_sq+C_squeeze**2 # add square of coefficient
            n_new=i_photon+1 # value of n to be used in 7.60 for next term in the recursion
            C_squeeze=-nu_over_mu*np.sqrt(n_new/(n_new+1))*C_squeeze # apply recursion relationship from 7.60
    norm_psi=np.sqrt(sum(abs(psi_out)**2))
    psi_out=psi_out/norm_psi # normalise output wavefunction
    return psi_out
    
def make_squeezed_number_states_analytic(xi_abs,theta_squeeze,n_trunc): # makes a list of squeezed number states analytically (these can then be used to make squeezed thermal states analytically)
    '''
        xi_abs is the absolute magnitude of the squeezing
        theta_squeeze is the angle of the squeezing
        n_trunc is the maximum number of photons to allow
    '''
    a=QOtrunc.make_annihiltion_Op(n_trunc=n_trunc) # creation operator
    a_d=a.conj().T #creation operator
    psi=make_squeezed_vac_psi_analytic(xi_abs,theta_squeeze,n_trunc) # wavefunction from squeezing vacuum
    S_a_d_S_d=a_d*np.cosh(xi_abs)+a*np.exp(-1j*theta_squeeze)*np.sinh(xi_abs) # 7.38 of G+K with a_d instead of a
    psi_list=[]
    for i_photon in range(n_trunc+1):
        psi_list=psi_list+[copy.copy(psi)] # copy psi value into list
        psi=S_a_d_S_d@psi # apply transformed creation operator
        norm_psi=np.sqrt(sum(abs(psi)**2))
        psi=psi/norm_psi # ensure normalisation
    return psi_list
    
def make_squeezed_thermal_state_analytic(xi_abs,theta_squeeze,T_therm,n_trunc): # makes a squeezed thermal state based on analytic methods
    '''
        xi_abs is the magnitude of the squeeezing
        theta_squeeze is the squeezing angle
        T_therm is the temperature of the thermal state which is squeezed
        n_trunc is the maximum number of phtons allowed in the hilbert space
    '''
    psi_list_num=make_squeezed_number_states_analytic(xi_abs,theta_squeeze,n_trunc) # list of wavefunctions starting at different number states
    therm_probs=create_thermal_rho(T_therm,n_trunc,diagonal_only=True) # create an array of thermal probabilities
    rho_therm_squeeze_m=np.zeros([n_trunc+1,n_trunc+1])
    for i_n in range(n_trunc): # loop through photon numbers
        rho_m=vectorized_rho_to_matrix_rho(pure_state_to_vectorised_rho(psi_list_num[i_n]))
        rho_therm_squeeze_m=rho_therm_squeeze_m+therm_probs[i_n]*rho_m # add weighted squeezed number state
    return rho_therm_squeeze_m
    
def general_Gauss_rho(alpha_abs,theta_disp,xi_abs,theta_squeeze,T_therm,n_trunc): # build the most general single mode Gaussian mixed state
    '''
        alpha_abs is the magnitude of the displacement
        theta_disp is the angle of the displacement
        xi_abs is the magnitude of the squeezing
        theta_squeeze is the angle of the squeeezing
        T_therm is the temprature of an initial thermal state
        n_trunc is the maximum photon number to allow
        returns Gaussian density matrix as a vector
    '''
    rho_therm_squeeze_m=make_squeezed_thermal_state_analytic(xi_abs,theta_squeeze,T_therm,n_trunc) # state with squeezing and displacement
    dispH=1j*QOtrunc.apply_displacement_Op(alpha_abs*np.exp(1j*theta_disp),psi=np.zeros(n_trunc+1),n_before=1,n_after=1,gen_mat=True) # generating matrix
    disp_gen_supOp=build_coherent_supOp(dispH)
    rho_therm_squeeze_v=matrix_rho_to_vector_rho(rho_therm_squeeze_m) # vectorize
    rho_Gauss_v=supOP_time_evolution(rho_therm_squeeze_v,disp_gen_supOp,t=1) # full density matrix as a vector
    return rho_Gauss_v
    
def build_Gauss_approx(rho): # returns the density matrix of a general Gaussian state which matches the displacment and covariance of a given state
    '''
        rho is a density matrix in either vectorized or array form, it does not need to be Gaussian
        returns rho_g_m, a Gaussian matrix which matches (as closely as possible) the displacement and covariance
    '''
    rho_m=convert_rho_to_matrix_idem(rho) # idempotent conversion fuction
    n_trunc=rho_m.shape[0]-1 # make final density matrix the same size
    (d_arr,sigma_arr)=calculate_displacement_and_covariance_single_mode_mixed(rho_m)
    (xi_abs,theta_squeeze,T_therm)=extract_Gaussian_state_parameters(sigma_arr) # extract parameters related to squeezing
    alpha_abs=np.sqrt(d_arr[0]**2+d_arr[1]**2) # length of displacement vector
    theta_disp=np.arctan(d_arr[1]/d_arr[0]) # calculate displacement angle
    if np.isnan(theta_disp):
        theta_disp=0 # special case for nan value
    if np.isnan(theta_squeeze):
        theta_squeeze=0 # special case for nan value
    if np.isnan(T_therm):
        T_therm=0 # special case for nan value
    if d_arr[0]<0: # if this element is negative
        theta_disp=theta_disp+np.pi # shift by pi
    rho_Gauss_v=general_Gauss_rho(alpha_abs,theta_disp,xi_abs,theta_squeeze,T_therm,n_trunc)
    return rho_Gauss_v
    
def Gauss_approx_stats(rho): # calculates a Gaussian approximation of a density matrix rho as well as the general statistics
    '''
        rho is a density matrix in either vectorized or array form, it does not need to be Gaussian
        returns fidelity with Gaussian approximation, excess photons compared to Gaussian approximation, and von Neumann entropy deficit compared to Gaussian approximation
    '''
    rho_m=convert_rho_to_matrix_idem(rho) # idempotent conversion fuction
    rho_Gauss_m=vectorized_rho_to_matrix_rho(build_Gauss_approx(rho)) # Gaussian approximation density matrix as an array
    fidelity_G=calculate_Op_expectation(rho_m,rho_Gauss_m) # trace of the product is fidelity
    n_trunc=rho_m.shape[0]-1 # maximum number of photons in Hilbert space
    a=QOtrunc.make_annihiltion_Op(n_trunc=n_trunc) # annihilation operator
    a_d=a.conj().T # creation operators
    n_bar_diff=calculate_Op_expectation(rho_m,a_d@a)-calculate_Op_expectation(rho_Gauss_m,a_d@a) # number expectation minus number expecatation of Gaussian approximation
    svn_Gauss=calculate_von_Neumann_entropy(rho_Gauss_m) # von Neumann entropy of Gaussian approximation
    svn_deficit=(svn_Gauss-calculate_von_Neumann_entropy(rho_m)) # entropy in Gaussian approximation minus entropy in the actual state(note this is the opposite way around as number difference, and should be positive because thermal distribution maximises for a given energy
    return (fidelity_G,n_bar_diff,svn_deficit)
    
def analytical_Weyl_charac_2x2(lam,rho_2): # finds the analytical value of the Weyl ordered charactoristic function given a 2x2 density matrix rho_2
    '''
        lam is the complex value of lammbda
        rho_2 is a 2x2 density matrix as ether an array or vector
    '''
    rho_2_m=convert_rho_to_matrix_idem(rho_2)
    C_w_val=(rho_2_m[0,0]+(1-abs(lam)**2)*rho_2_m[1,1]+2*1j*np.imag(lam*rho_2_m[0,1]))*np.exp(-0.5*abs(lam)**2) # analytical formula (see paper)
    return C_w_val

def brute_force_Weyl_charac(rho,lam1_range=[-5,5],lam2_range=[-5,5],lam1_grid_points=100,lam2_grid_points=100): # calculates the Weyl-ordered charactoristic function at every point on a grid rather than using underlying structure
    '''
        rho is a density matrix in either matrix or vector form
        lam1_range is the range of real lambda values values
        lam2_range is the range of imaginary lamba values
        lam1_grid_points is the number of real lambda grid points
        lam2_grid_points is the number of imaginary lambda grid points
        returns an lam1_grid_points by lam2_grid_points array which are the Weyl-ordered charactoristic function, equation 3.128a in G+K as well as the grid values themselves
    '''
    rho_m=convert_rho_to_matrix_idem(rho) # idempotent conversion fuction
    lam1_val_array=np.linspace(lam1_range[0],lam1_range[1],lam1_grid_points) # array of X1 values
    lam2_val_array=np.linspace(lam2_range[0],lam2_range[1],lam2_grid_points) # array of X2 values
    Weyl_charac_array=np.zeros([lam1_grid_points,lam2_grid_points],dtype=complex) # make array for storing Weyl-ordered charactoristic functions
    if rho_m.shape[0]==2: # if rho is a 2x2 matrix then use analytical definition of Weyl charactoristic
        for ilam1 in range(lam1_grid_points):
            for ilam2 in range(lam2_grid_points):
                lam=lam1_val_array[ilam1]+1j*lam2_val_array[ilam2] #lambda
                Weyl_charac_array[ilam1,ilam2]=analytical_Weyl_charac_2x2(lam,rho_m)
    else:
        for ilam1 in range(lam1_grid_points):
                for ilam2 in range(lam2_grid_points):
                    lam=lam1_val_array[ilam1]+1j*lam2_val_array[ilam2] #lambda
                    dispOp=QOtrunc.apply_displacement_Op(lam,psi=np.zeros(rho_m.shape[0]),n_before=1,n_after=1,gen_mat=True) # displacement operator
                    Weyl_charac_array[ilam1,ilam2]=calculate_Op_expectation(rho_m,linalg_d.expm(dispOp.todense()))
    return [Weyl_charac_array,lam1_val_array,lam2_val_array]

def make_2D_FT_array(lam,X1_range=[-5,5],X2_range=[-5,5],X1_grid_points=100,X2_grid_points=100): # matrix of complex values for 2D Fourier transform
    '''
        lam is a complex number corresponding to the value of lambda
        X1_range is the range of the first quadrature in the final plot
        X2_range is the range of the second quadrature in the final plot
        X1_grid_points is the number of grid points in the first quadrature in the final plot
        X2_grid_points is the number of grid points in the second quadrature in the final plot
    '''
    real_alpha_component=np.outer(np.linspace(X1_range[0],X1_range[1],X1_grid_points),np.ones([1,X2_grid_points],dtype=complex)) # real component of alpha by coordinate counstructed via outer product
    im_alpha_component=np.outer(np.ones([X1_grid_points,1],dtype=complex),np.linspace(X2_range[0],X2_range[1],X2_grid_points)) # imaginary component of alpha by coordinate constructed via outer product
    alpha_val_array=real_alpha_component+1j*im_alpha_component # full array of complex values
    FT_phase_array=np.exp(np.conjugate(lam)*alpha_val_array-lam*alpha_val_array.conj()) # calculate phase array (note elementwise exponentiation not matrix exponentiation) from equation 3.136, 3.133 of G+K
    return FT_phase_array
    
def inverse_FT_charac(charac,lam1_val_array,lam2_val_array,X1_range=[-5,5],X2_range=[-5,5],X1_grid_points=100,X2_grid_points=100): # performs an inverse 2D Fourier transform to move from charactoristic function to quadrature space distribution
    '''
        charac is the charactoristic function to be transformed as an array
        lam1_val_array is the grid point array for the real part of lambda
        lam2_val_array is the grid point array for the imaginary part of lambda
        X1_range is the range of the first quadrature
        X2_range is the range of the second quadrature
        X1_grid_points is the number of grid points for the first quadrature
        X2_grid_points is the number of grid points for the second quadrature
    '''
    final_distribution_array=np.zeros([X1_grid_points,X2_grid_points],dtype=complex) #make an array for the final distribution
    for ilam1 in range(len(lam1_val_array)): # loop through real parts of lambda
        for ilam2 in range(len(lam2_val_array)): # loop through imaginary parts of lambda
            lam=lam1_val_array[ilam1]+1j*lam2_val_array[ilam2] # complex value of lambda
            if np.abs(lam)>10**-15: # don't bother if lambda is too small
                lamda_FT_component=make_2D_FT_array(lam,X1_range=X1_range,X2_range=X2_range,X1_grid_points=X1_grid_points,X2_grid_points=X2_grid_points) # calculate Fourier component
                final_distribution_array=final_distribution_array+charac[ilam1,ilam2]*lamda_FT_component # add to final array
    area_element=(lam1_val_array[1]-lam1_val_array[0])*(lam2_val_array[1]-lam2_val_array[0]) # assumes a uniform grid
    final_distribution_array=final_distribution_array*area_element/(np.pi**2) # normalise
    X1_vals=np.linspace(X1_range[0],X1_range[1],X1_grid_points)
    X2_vals=np.linspace(X2_range[0],X2_range[1],X2_grid_points)
    return [final_distribution_array,X1_vals,X2_vals]
    
def make_Gaussian_Q_calc(lam1_val_array,lam2_val_array): # makes a Gaussian to allow calculation of the Q function from the Wigner-ordered charactoristic
    '''
        lam1_vals_array are values of the real part of lambda
        lam2_val_array are values of the imaginary part of lambda
    '''
    abs_lam_sq_array=abs(np.outer(lam1_val_array,np.ones(len(lam2_val_array),dtype=complex)))**2+abs(np.outer(np.ones(len(lam1_val_array),dtype=complex),lam1_val_array))**2 # sum of squared vales
    Gauss_function=np.exp(-0.5*abs_lam_sq_array) # Gaussian function following eq 3.129 of G+K, note elementwise exponentiation not matrix exponentiation
    return Gauss_function
    
def analytical_Wigner_function_2x2(alpha,rho_2): # finds the analytical value of the Wigner function given a 2x2 density matrix rho_2
    '''
        alpha is the complex value of alpha
        rho_2 is a 2x2 density matrix as ether an array or vector
    '''
    rho_2_m=convert_rho_to_matrix_idem(rho_2)
    Wig_val=(2/np.pi)*(rho_2_m[0,0]+(4*(abs(alpha)**2)-1)*rho_2_m[1,1]+4*np.real(alpha*rho_2_m[0,1]))*np.exp(-2*abs(alpha)**2) # analytical formula (see paper)
    return Wig_val
    
def analytical_Q_function_2x2(alpha,rho_2): # finds the analytical value of the Q function given a 2x2 density matrix rho_2
    '''
        alpha is the complex value of alpha
        rho_2 is a 2x2 density matrix as ether an array or vector
    '''
    rho_2_m=convert_rho_to_matrix_idem(rho_2)
    Q_val=(1/np.pi)*(rho_2_m[0,0]+((abs(alpha)**2))*rho_2_m[1,1]+2*np.real(alpha*rho_2_m[0,1]))*np.exp(-abs(alpha)**2) # analytical formula (see paper)
    return Q_val
    
def analytical_Wigner_vals_2x2(rho_2,X1_range=[-5,5],X2_range=[-5,5],X1_grid_points=100,X2_grid_points=100): # analytical formula for Wigner function for 2x2 density matrix
    '''
        rho_2 is a 2x2 density matrix in either vector or array form
        X1_range is the range of the first quadrature in the final plot
        X2_range is the range of the second quadrature in the final plot
        X1_grid_points is the number of grid points in the first quadrature in the final plot
        X2_grid_points is the number of grid points in the second quadrature in the final plot
        returns the grid points and the Wigner function values
    '''
    rho_m=convert_rho_to_matrix_idem(rho_2) # idempotent conversion fuction
    X1_vals=np.linspace(X1_range[0],X1_range[1],X1_grid_points) # array of X1 values
    X2_vals=np.linspace(X2_range[0],X2_range[1],X2_grid_points) # array of X2 values
    Wigner_vals=np.zeros([X1_grid_points,X2_grid_points],dtype=complex) # make array for storing Weyl-ordered charactoristic functions
    if rho_m.shape[0]==2: # if rho is a 2x2 matrix then use analytical definition of Weyl charactoristic
        for iX1 in range(X1_grid_points):
            for iX2 in range(X2_grid_points):
                alpha=X1_vals[iX1]+1j*X2_vals[iX2] # alpha
                Wigner_vals[iX1,iX2]=analytical_Wigner_function_2x2(alpha,rho_m)
    return [Wigner_vals,X1_vals,X2_vals]
    
def analytical_Q_vals_2x2(rho_2,X1_range=[-5,5],X2_range=[-5,5],X1_grid_points=100,X2_grid_points=100): # analytical formula for Q function for 2x2 density matrix
    '''
        rho_2 is a 2x2 density matrix in either vector or array form
        X1_range is the range of the first quadrature in the final plot
        X2_range is the range of the second quadrature in the final plot
        X1_grid_points is the number of grid points in the first quadrature in the final plot
        X2_grid_points is the number of grid points in the second quadrature in the final plot
        returns the grid points and the Q function values
    '''
    rho_m=convert_rho_to_matrix_idem(rho_2) # idempotent conversion fuction
    X1_vals=np.linspace(X1_range[0],X1_range[1],X1_grid_points) # array of X1 values
    X2_vals=np.linspace(X2_range[0],X2_range[1],X2_grid_points) # array of X2 values
    Q_vals=np.zeros([X1_grid_points,X2_grid_points],dtype=complex) # make array for storing Q values
    if rho_m.shape[0]==2: # if rho is a 2x2 matrix then use analytical definition of Weyl charactoristic
        for iX1 in range(X1_grid_points):
            for iX2 in range(X2_grid_points):
                alpha=X1_vals[iX1]+1j*X2_vals[iX2] # alpha
                Q_vals[iX1,iX2]=analytical_Q_function_2x2(alpha,rho_m)
    return [Q_vals,X1_vals,X2_vals]
    

def find_Wigner_and_Q_brute_force_charac(rho,charac_calc_params={}): # calculates Wigner and Q function based on brute-force charactoristic funciton calculations
    '''
        rho is the density matrix in either vector or array representation
        charac_calc_params is a dictionary of parameters (too avoid a large number of individual keywords, fields which are not present are set to defaults)
        set as empty dictionary wich takes all default values
            lam1_range is the range for the real parts of lambda defaults to [-5,5]
            lam2_range is the range for the imaginary parts of lambda defaults to [-5,5]
            lam1_grid_points is the number of grid points used for the real part of lambda, defaults to 100
            lam2_grid_points is the number of grid points used for the imaginary part of lambda, defaults to 100
            X1_range is the range for the first quadrature defaults to [-5,5]
            X2_range is the range for the second quadrature defaults to [-5,5]
            X1_grid_points is the number of grid points used for the first quadrature, defaults to 100
            X2_grid_points is the number of grid points used for the second quadrature, defaults to 100
        returns Wigner and Q functions as well as a dictionary of additional data
    '''
    if 'lam1_range' in charac_calc_params:
        lam1_range=charac_calc_params['lam1_range']
    else:
        lam1_range=[-5,5]
    if 'lam2_range' in charac_calc_params:
        lam2_range=charac_calc_params['lam2_range']
    else:
        lam2_range=[-5,5]
    if 'lam1_grid_points' in charac_calc_params:
        lam1_grid_points=charac_calc_params['lam1_grid_points']
    else:
        lam1_grid_points=100
    if 'lam2_grid_points' in charac_calc_params:
        lam2_grid_points=charac_calc_params['lam2_grid_points']
    else:
        lam2_grid_points=100
    if 'X1_range' in charac_calc_params:
        X1_range=charac_calc_params['X1_range']
    else:
        X1_range=[-5,5]
    if 'X2_range' in charac_calc_params:
        X2_range=charac_calc_params['X2_range']
    else:
        X2_range=[-5,5]
    if 'X1_grid_points' in charac_calc_params:
        X1_grid_points=charac_calc_params['X1_grid_points']
    else:
        X1_grid_points=100
    if 'X2_grid_points' in charac_calc_params:
        X2_grid_points=charac_calc_params['X2_grid_points']
    else:
        X2_grid_points=100
    rho_m=convert_rho_to_matrix_idem(rho) # idempotent conversion fuction
    [Weyl_charac_array,lam1_val_array,lam2_val_array]=brute_force_Weyl_charac(rho_m,lam1_range=lam1_range,lam2_range=lam2_range,lam1_grid_points=lam1_grid_points,lam2_grid_points=lam2_grid_points)
    Gauss1=make_Gaussian_Q_calc(lam1_val_array,lam2_val_array) # Gaussian function to allow calculation of Husimi-Q
    Q_charac_array=Weyl_charac_array*Gauss1 # note elementwise not matrix multiplication
    if rho_m.shape[0]==2: # use analytical formula for 2x2 version
        [Wigner_vals,X1_vals,X2_vals]=analytical_Wigner_vals_2x2(rho_m,X1_range=X1_range,X2_range=X2_range,X1_grid_points=X1_grid_points,X2_grid_points=X2_grid_points)
        [Q_vals,X1_vals,X2_vals]=analytical_Q_vals_2x2(rho_m,X1_range=X1_range,X2_range=X2_range,X1_grid_points=X1_grid_points,X2_grid_points=X2_grid_points)
    else:
        [Wigner_vals,X1_vals,X2_vals]=inverse_FT_charac(Weyl_charac_array,lam1_val_array,lam2_val_array,X1_range=X1_range,X2_range=X2_range,X1_grid_points=X1_grid_points,X2_grid_points=X2_grid_points)
        [Q_vals,X1_vals,X2_vals]=inverse_FT_charac(Q_charac_array,lam1_val_array,lam2_val_array,X1_range=X1_range,X2_range=X2_range,X1_grid_points=X1_grid_points,X2_grid_points=X2_grid_points)
    neg_volume=np.sum(abs(np.real(Wigner_vals)))*((X1_vals[1]-X1_vals[0])*(X2_vals[1]-X2_vals[0]))-1 # assumes an equally spaced grid, measure of quantum behaviour from arXiv:quant-ph/0406015
    additional_data={'neg_volume':neg_volume,'Weyl_charac_array':Weyl_charac_array,'Q_charac_array': Q_charac_array,'lam1_val_array':lam1_val_array,'lam2_val_array':lam2_val_array} # charactoristic function data and negative volume
    return [Wigner_vals,Q_vals,X1_vals,X2_vals,additional_data]
    
    
def even_odd_projectors(n_trunc=100): # builds projectors to even and odd photon sectors
    '''
        n_trunc is the number of photons
        returns projectors as sparse matrices
    '''
    even_rows=list(range(0,n_trunc+1,2))
    even_columns=list(range(0,len(even_rows)))
    proj_array_even=sparse.csr_array((np.ones(len(even_rows)),(even_rows,even_columns)),shape=(n_trunc+1,len(even_rows))) # sparse matrix to project out only even components
    odd_rows=list(range(1,n_trunc+1,2))
    odd_columns=list(range(0,len(odd_rows)))
    proj_array_odd=sparse.csr_array((np.ones(len(odd_rows)),(odd_rows,odd_columns)),shape=(n_trunc+1,len(odd_rows))) # sparse matrix to project out only odd components
    return(proj_array_even,proj_array_odd)
    
def evo_gen_diag_two_photon_lindblad(diag_length,diag_index=1,n_before=1,n_after=1): # makes a sparse matrix which generates the Lindblad dynamics for just one diagonal under two photon loss
    '''
        diag_length is how many elements of the diagonal to generate
        diag_index is the index of the diagonal which is to be evolved, diag_index=0 takes the main diagonal, diag_index=1 takes the first off diagonal, -1 takes a lower dagonal, etc..., defaults to the first off-diagonal
        n_before is the size of an identity matrix tensored before
        n_after is the size of an idendity matrix tensored after
    '''
    evo_gen_rows=[]
    evo_gen_columns=[]
    evo_gen_entries=[]
    for i_pop_evo in range(diag_length): # entries and locations of evolution generators
        evo_gen_rows=evo_gen_rows+[i_pop_evo] # first index for diagonal
        evo_gen_columns=evo_gen_columns+[i_pop_evo] # second index for diagonal
        evo_gen_entries=evo_gen_entries+[-0.5*i_pop_evo*(i_pop_evo-1)-0.5*(i_pop_evo+abs(diag_index)-1)*(i_pop_evo+abs(diag_index))] # add diagonal entry corresponding to anti-commutatator component of Lindblad equation
        #evo_gen_entries=evo_gen_entries+[-0.5*i_pop_evo*(i_pop_evo-1)-0.5*(i_pop_evo+diag_index-1)*(i_pop_evo+diag_index)] # add diagonal entry corresponding to anti-commutatator component of Lindblad equation
        if i_pop_evo>1:
            evo_gen_rows=evo_gen_rows+[i_pop_evo-2] # first index for off diagonal
            evo_gen_columns=evo_gen_columns+[i_pop_evo] # second index for off diagonal
            evo_gen_entries=evo_gen_entries+[np.sqrt(i_pop_evo*(i_pop_evo-1)*(i_pop_evo+abs(diag_index))*(i_pop_evo+abs(diag_index)-1))] # new entry component corresponding to first part of Lindblad equation
            #evo_gen_entries=evo_gen_entries+[np.sqrt(i_pop_evo*(i_pop_evo-1)*(i_pop_evo+diag_index)*(i_pop_evo+diag_index-1))] # new entry component corresponding to first part of Lindblad equation
    evo_gen_diag=sparse.csr_array((evo_gen_entries,(evo_gen_rows,evo_gen_columns)),shape=(diag_length,diag_length)) # make sparse matrix
    evo_gen_diag=QOtrunc.tensor_identities(evo_gen_diag,n_before,n_after) # tensor in identities as needed
    return evo_gen_diag
    
def calculate_diagonal_from_pure_state(psi0,diag_index=1): # calculates a diagonal of the density matrix form a pure state
    '''
        psi0 is an initial pure state
        diag_index is the index of the diagonal which is to be evolved, diag_index=0 takes the main diagonal, diag_index=1 takes the first off diagonal, etc..., defaults to the first off-diagonal
    '''
    diag_vec0=calculate_diagonal_from_pure_states(psi0,psi0,diag_index=diag_index)
    return diag_vec0
    
def calculate_diagonal_from_pure_states(psi0,psi1,diag_index=1): # calculates a diagonal of the density matrix form a pair of pure state
    '''
        psi0 is an initial pure state
        psi1 is another pure state, possibly identical to psi0
        diag_index is the index of the diagonal which is to be evolved, diag_index=0 takes the main diagonal, diag_index=1 takes the first off diagonal, etc..., defaults to the first off-diagonal negative numbers take lower diagonals
    '''
    if not diag_index<0:
        diag_vec0=psi0[:(len(psi0)-diag_index)]*np.conj(psi1[diag_index:]) # vector of diagonal elements
    else: # special case for lower diagonals
        diag_vec0=psi0[abs(diag_index):]*np.conj(psi1[:(len(psi0)-abs(diag_index))]) # vector of diagonal elements
    return diag_vec0
    
def two_photon_loss_diag_lindblad_evolve(diag_vec0,diag_index=1,gammat=30,n_before=1,n_after=1):
    '''
        diag_vec0 is the initial diagonal vector
        diag_index is the index of the diagonal which is to be evolved, diag_index=0 takes the main diagonal, diag_index=1 takes the first off diagonal, etc..., defaults to the first off-diagonal
        gammat is the product of time and the decay rate, defaults to 20 
        n_before is the size of an identity matrix tensored before
        n_after is the size of an identity matrix tensored after
        returns the final off diagonal
    '''
    
    diag_length=len(diag_vec0)
    subspace_size=int(round(diag_length/(n_before*n_after)))
    evo_gen_diag=evo_gen_diag_two_photon_lindblad(subspace_size,diag_index=diag_index,n_before=n_before,n_after=n_after) # make generator matrix
    if diag_length>3001: # the sparse method is slower for small matrices
        diag_vec=linalg.expm_multiply(gammat*evo_gen_diag,diag_vec0) # apply Lindblad superoperator
    else:
        gen_exp=linalg_d.expm(gammat*evo_gen_diag.todense()) # dense matrix exponentiation
        diag_vec=gen_exp@diag_vec0 # multiply the densly calculated version
    return diag_vec

def two_photon_loss_zero_one_density_matrix_component_from_diags(main_diag0,off_diag0_up,off_diag0_low,gammat=30): # uses structure of density matrix to calculate only the zero and one photon subspace without having to construct the full matrix
    '''
        main_diag is the main diagonal of the density matrix can be set to None for off-diagonal only computation
        off_diag_up is the upper off diagonal of the density matrix can be set to None for diagonal only computation
        off_diag_low is the upper off diagonal of the density matrix can be set to None for diagonal only computation
        gammat is the loss rate multipled by time
    '''
    rho_trunc_m=np.zeros([2,2],dtype=complex) # empty 2x2 array
    if not type(main_diag0) is type(None):
        main_diag=two_photon_loss_diag_lindblad_evolve(main_diag0,diag_index=0,gammat=gammat)
        rho_trunc_m[0,0]=main_diag[0]
        rho_trunc_m[1,1]=main_diag[1]
    if not type(off_diag0_up) is type(None):
        off_diag=two_photon_loss_diag_lindblad_evolve(off_diag0_up,diag_index=1,gammat=gammat)
        rho_trunc_m[0,1]=off_diag[0]
        #rho_trunc_m[1,0]=np.conj(off_diag[0])
    if not type(off_diag0_low) is type(None):
        off_diag=two_photon_loss_diag_lindblad_evolve(off_diag0_low,diag_index=-1,gammat=gammat)
        rho_trunc_m[1,0]=off_diag[0]
    return(rho_trunc_m)
    
def two_photon_loss_zero_one_density_matrix_component_from_pure_state(psi0,gammat=30):
    '''
        psi0 is an initial pure state 
        gammat is the loss rate multipled by time
    '''
    rho_trunc_m=two_photon_loss_zero_one_density_matrix_component_from_pure_states(psi0,psi0,gammat=30)
    return(rho_trunc_m)
    
def two_photon_loss_zero_one_density_matrix_component_from_pure_states(psi0,psi1,gammat=30):
    '''
        psi0 is an initial pure state
        psi1 is a separate, possibly identical pure state 
        gammat is the loss rate multipled by time
    '''
    main_diag0=calculate_diagonal_from_pure_states(psi0,psi1,diag_index=0)
    off_diag0_up=calculate_diagonal_from_pure_states(psi0,psi1,diag_index=1)
    off_diag0_low=calculate_diagonal_from_pure_states(psi0,psi1,diag_index=-1)
    rho_trunc_m=two_photon_loss_zero_one_density_matrix_component_from_diags(main_diag0,off_diag0_up,off_diag0_low,gammat=gammat)
    return(rho_trunc_m)
    
def make_2x2_rho(zeta,eta,theta=0): # makes an arbitrary (valid) two by two density matrix assuming only the vacuum and single photon states are occupied
    '''
        zeta is the probability to be in |1>, valid range between 0 and 1
        eta is a multiplicative factor on the coherence, ranges between 0 (incoherent superposition) and 1 (fully coherent superposition)
        theta is the phase on the off diagonal element, any real number is allowed
        returns the matrix
    '''
    rho_2=np.zeros([2,2],dtype=complex) # will generally have complex entries
    rho_2[0,0]=1-zeta
    rho_2[1,1]=zeta
    rho_2[0,1]=eta*np.sqrt(zeta*(1-zeta))*np.exp(1j*theta)
    rho_2[1,0]=eta*np.sqrt(zeta*(1-zeta))*np.exp(-1j*theta)
    return rho_2
    
def apply_displacement_Op_rho(alpha,rho,n_before=1,n_after=1,gen_mat=False): # applies a displacement operator in a density matrix setting
    '''
        alpha is a complex number which gives the amount of displacement
        rho is the input density matrix in any form, or the size of a vectorized density matrix if asked to return a generator
        n_before is the linear size of the density matrix subspace which comes before the displacement (defaults to 1)
        n_after is the linear size of the density matrix subspace which comes before the displacement (defaults to 1)
        gen_mat is a flag which returns the generator rather than applying the evolutions
    '''
    rho_v=convert_rho_to_vector_idem(rho) # idempotent conversion fuction
    n_photon_max=get_subspace_photons_rho(rho_v,n_before=n_before,n_after=n_after) # maximum number of photons in subspace
    disp_gen=QOtrunc.apply_displacement_Op(alpha,n_photon_max+1,n_before=1,n_after=1,gen_mat=True) # make generator matrix
    if gen_mat: # if asked for the generator matrix rather than the time evolution
        disp_gen_sup=build_coherent_supOp(1j*disp_gen) # note factor of 1j needed for construction
        supOp=QOtrunc.tensor_identities(H_disp,n_before=n_before**2,n_after=n_after**2) # tensor in additional subspaces, note square of linear sizes
        return supOp
    else: # otherwise apply time evolution
        supOp_U_gen=psi_gen_to_supOp(disp_gen) # produce the superoperator which applies the unitary
        supOp_U=QOtrunc.tensor_identities(supOp_U_gen,n_before=n_before**2,n_after=n_after**2) # tensor in additional subspaces, note square of linear sizes
        rho_v_f=supOp_U@rho_v # apply superoperator to vectorized density matrix
        return rho_v_f
        
def apply_squeezing_Op_rho(xi,rho,n_before=1,n_after=1,gen_mat=False): # applies a squeezing operator in a density matrix setting
    '''
        xi is a complex number whih gives the amount of ssqueezing
        rho is the input density matrix in any form, or the size of a vectorized density matrix if asked to return a generator
        n_before is the linear size of the density matrix subspace which comes before the displacement (defaults to 1)
        n_after is the linear size of the density matrix subspace which comes before the displacement (defaults to 1)
        gen_mat
    '''
    rho_v=convert_rho_to_vector_idem(rho) # idempotent conversion fuction
    n_photon_max=get_subspace_photons_rho(rho_v,n_before=n_before,n_after=n_after) # maximum number of photons in subspace
    squeeze_gen=QOtrunc.apply_squeezing_Op(xi,n_photon_max,n_before=1,n_after=1,gen_mat=True) # make generator matrix
    if gen_mat: # if asked for the generator matrix rather than the time evolution
        squeeze_gen_sup=build_coherent_supOp(1j*squeeze_gen) # note factor of 1j needed for construction
        supOp=QOtrunc.tensor_identities(squeeze_gen_sup,n_before=n_before,n_after=n_after) # tensor in additional subspaces
        return supOp
    else: # otherwise apply time evolution
        supOp_U_gen=psi_gen_to_supOp(squeeze_gen) # produce the superoperator which applies the unitary
        supOp_U=QOtrunc.tensor_identities(supOp_U_gen,n_before=n_before**2,n_after=n_after**2) # tensor in additional subspaces, note square of linear sizes
        rho_v_f=supOp_U@rho_v # apply superoperator to vectorized density matrix
        return rho_v_f
        
def apply_beamsplitting_Op_rho(theta,rho,n_before,n_after,gen_mat=False): # applies a beamsplitter to a mixed state represented by a density matrix
    '''
        theta is a real parameter which defines the degree of beamsplitting which is applied following an extension of the definition of beamsplitting given in equation 6.12 of Gerry and Knight such that theta=pi/4 corresponds to a 50:50 beamsplitter 
        rho is the density matrix (or the size of the state vector if gen_mat=True)
        n_before and n_after are lists of the number before and after for each mode such that n_before[0] and n_after[0] correspond to the first mode and n_before[1] and n_after[1] correspond to the second mode note these are linear sizes
        returns the density matrix if gen_mat==False and the generating sparse matrix if gen_mat==True
    '''
    rho_v=convert_rho_to_vector_idem(rho) # idempotent conversion fuction
    psi_len=int(np.sqrt(np.round(len(rho_v)))) # the total length of the state vector
    beamsplit_gen=QOtrunc.apply_beamsplitting_Op(theta,psi_len,n_before,n_after,gen_mat=True) # build the generating matrix for beamsplitting on a state vector
    if gen_mat: # if asked for the generator matrix rather than the time evolution
        supOp=build_coherent_supOp(1j*beamsplit_gen) # note factor of 1j needed for construction
        return supOp
    else: # otherwise apply time evolution
        supOp_U=psi_gen_to_supOp(squeeze_gen) # produce the superoperator which applies the unitary
        rho_v_f=supOp_U@rho_v # apply superoperator to vectorized density matrix
        return rho_v_f
    

def zero_one_photon_subspace_displace_squeeze_loss_supOp(n_hat_disp=0,n_hat_squeeze=0,n_sub=31,phi_disp=0,phi_squeeze=0): # computes the superoperator in the |0> |1> subspace for displacement and squeezing followed by strong two-photon loss to return to the subspace
    '''
        n_hat_disp is the number of photons from displacement
        n_hat_squeeze is the number of photons from squeezing
        n_sub is the total size of the expanded subspace used in the calculation
        phi_disp is the angle of the displacement
        phi_squeeze is the angle of the squeezing
    '''
    psi_n0=QOtrunc.prepare_squeezed_state_photon_number(n_hat_disp=n_hat_disp,n_hat_squeeze=n_hat_squeeze,n_sub=n_sub,phi_disp=phi_disp,phi_squeeze=phi_squeeze,start_photons=0) # state vector for squeezing and displacement starting with no photons
    psi_n1=QOtrunc.prepare_squeezed_state_photon_number(n_hat_disp=n_hat_disp,n_hat_squeeze=n_hat_squeeze,n_sub=n_sub,phi_disp=phi_disp,phi_squeeze=phi_squeeze,start_photons=1) # state vector for squeezing and displacement starting with one photon
    supOp_subspace=np.zeros([4,4],dtype=complex) # new superoperator
    rho_m_new=two_photon_loss_zero_one_density_matrix_component_from_pure_state(psi_n0,gammat=30) # new rho from 00 element as a matrix
    supOp_subspace[:,0]=matrix_rho_to_vector_rho(rho_m_new) # convert to a vector and add to superoperator
    rho_m_new=two_photon_loss_zero_one_density_matrix_component_from_pure_states(psi_n0,psi_n1,gammat=30) # new rho from 01 element as a matrix
    supOp_subspace[:,1]=matrix_rho_to_vector_rho(rho_m_new) # convert to a vector and add to superoperator
    #supOp_subspace[:,2]=0.5*matrix_rho_to_vector_rho(rho_m_new)
    rho_m_new=two_photon_loss_zero_one_density_matrix_component_from_pure_states(psi_n1,psi_n0,gammat=30) # new rho from 10 element as a matrix
    #supOp_subspace[:,1]=supOp_subspace[:,1]+0.5*matrix_rho_to_vector_rho(rho_m_new) # convert to a vector and add to superoperator
    #supOp_subspace[:,2]=supOp_subspace[:,2]+0.5*matrix_rho_to_vector_rho(rho_m_new) # convert to a vector and add to superoperator
    supOp_subspace[:,2]=matrix_rho_to_vector_rho(rho_m_new) # convert to a vector and add to superoperator
    rho_m_new=two_photon_loss_zero_one_density_matrix_component_from_pure_state(psi_n1,gammat=30) # new rho from 00 element as a matrix
    supOp_subspace[:,3]=matrix_rho_to_vector_rho(rho_m_new) # convert to a vector and add to superoperator
    return supOp_subspace
    
def apply_phase_shift_zero_one_supOp(phi): # applies a phase shift within the |0> |1> subspace
    '''
        phi is the angle of phase to apply
    '''
    H=1j*QOtrunc.apply_time_evolution_Op(phi,2,n_before=1,n_after=1,gen_mat=True).todense() # construct Hamiltonian for phase shifting
    supOp=linalg.expm(build_coherent_supOp(H).todense()) # build superoperator for Hamiltonian evolution to apply phase
    supOp=sparse.csr_array(supOp) # convert to sparse matrix for consistency
    return supOp

def find_fixed_point(supOp,start_rho=np.array([1,0,0,0]),err_tol=10**-10,max_app=40): # finds the fixed point of a superoperator up to a tolerance given by err_tol
    '''
        supOp is a superoperator
        start_rho is the state used at the beginning of the fixed point search defaults to the ground state of a two level system
        err_tol is the tolerance for error in the search defaults to 10**-10
        max_app is the number of times to apply powers of the matrix, corresponds to 2**max_app applications of the superoperator, defaults to 20, which corresonds to applying slightly over a trillion times
    '''
    start_rho_v=convert_rho_to_vector_idem(start_rho) # idempotent conversion fuction
    apply_supOp=copy.copy(supOp) # copy the superoperator to apply
    for i_app in range(max_app): # loop through applications of the superoperator
        fp_trial=apply_supOp@start_rho_v # construct trial state by applying a power of the superoperator to starting density matrix
        fp_trial_app=supOp@fp_trial # apply once more
        diff_vec=fp_trial-fp_trial_app # difference between the two vectors
        norm_diff=np.sqrt(sum(abs(diff_vec)**2)) # difference in the norms
        apply_supOp=apply_supOp@apply_supOp # effectively double the number of applications
        if norm_diff<err_tol: # if the density matrix is a fixed point up to error tolerances
            return fp_trial # return the trial fixed point
    raise RuntimeError('failed to converge to a fixed point')


def make_partial_shift_mask(n_sub,n_before=1,n_after=1,shift_number=1): # makes an effective mask of booleans which peform a shift to calculate off-diagonals
    '''
        n_sub is the size of the subspace
        n_before is the size of the previous subspace
        n_after is the size of the subspace after
    '''
    mask_up=np.array([True]*n_sub) # initially all true
    if shift_number<0: # do opposite for negative shift number
        mask_up[shift_number:]=False
    elif shift_number>0:
        mask_up[:shift_number]=False
    mask_up=np.kron(mask_up,np.ones(n_before,dtype=bool))
    mask_up=np.kron(np.ones(n_after,dtype=bool),mask_up)
    mask_dn=np.array([True]*n_sub) # initially all true
    if shift_number<0: # do opposite for negative shift number
        mask_dn[:(-shift_number)]=False
    elif shift_number>0:
        mask_dn[(-shift_number):]=False
    mask_dn=np.kron(mask_dn,np.ones(n_before,dtype=bool))
    mask_dn=np.kron(np.ones(n_after,dtype=bool),mask_dn)
    return(mask_up,mask_dn)
    
def partial_shifts_mask(subspace_sizes,subspace_diags): # builds a mask to perform multiple partial shifts, denoted by subspace_diags
    '''
        subspace_sizes are the sizes of each subspace
        subspace diags are which diagonals are taken in each subspace, with negative integers indicating lower diagonals
    '''
    mask_up_tot=np.array([True]) # start with single element
    mask_dn_tot=np.array([True]) # start with single element
    for i_sub in range(len(subspace_sizes)): # loop through subspaces
        (mask_up,mask_dn)=make_partial_shift_mask(subspace_sizes[i_sub],n_before=1,n_after=1,shift_number=subspace_diags[i_sub])
        mask_up_tot=np.kron(mask_up,mask_up_tot) # tensor product in new part of mask
        mask_dn_tot=np.kron(mask_dn,mask_dn_tot) # tensor product in new part of mask
    return(mask_up_tot,mask_dn_tot)
    
def compute_diag_psis(psi_ket,psi_bra,subspace_sizes,subspace_diags): # computes a given diagonal for a density matrix represented by the outer product of two state vectors, psi_bra and psi_ket
    '''
        psi_ket is the state vector which acts as a ket in the outer product
        psi_bra is the state vector which acts as a bra in the outer product
        subspace_sizes is a list of the sizes of the different subspaces
        subspace_diags is a list of which diagonal is taken within each subspace
    '''
    (mask_up_tot,mask_dn_tot)=partial_shifts_mask(subspace_sizes,subspace_diags) # make masks for computing diagonals
    diag_vec=psi_ket[mask_dn_tot]*(psi_bra[mask_up_tot].conj()) # note elementwise multiplication
    return diag_vec
    
    
def partial_trace_diag_matrix(n_sub,n_before,n_after): # builds a sparse matrix to perform a partial trace operation over a subspace of size n_sub with n_before and n_after
    '''
        n_sub is the size of the subspace to be traced over
        n_before is the total size of previous subspaces
        n_after is the total size of subsequent subspaces
    '''
    traceout_mat=sparse.csr_array(np.ones([1,n_sub])) # 1 by n_sub rectangular matrix to sum
    traceout_mat=sparse.kron(traceout_mat,sparse.eye_array(n_before))
    traceout_mat=sparse.kron(sparse.eye_array(n_after),traceout_mat)
    return traceout_mat
    
def partial_trace_matrix_list(n_sub,n_before,n_after): # returns a list of matrices which can be used to perform a trace over a subspace of size n_sub with n_before and n_after
    '''
        n_sub is the size of the subspace to be traced over
        n_before is the total size of previous subspaces
        n_after is the total size of subsequent subspaces
    '''
    traceout_mat_list=[]
    for i_sub in range(n_sub): # loop through values within the subspace
        base_array=np.zeros([1,n_sub])
        base_array[0,i_sub]=1 # one value only at a single location
        traceout_mat=sparse.csr_array(base_array) # 1 by n_sub rectangular matrix to isolate one diagonal elemetn
        traceout_mat=sparse.kron(traceout_mat,sparse.eye_array(n_before))
        traceout_mat=sparse.kron(sparse.eye_array(n_after),traceout_mat)
        traceout_mat_list=traceout_mat_list+[copy.copy(traceout_mat)] # copy the matrix and add to list
    return traceout_mat_list
    
def make_partial_trace_supOp_mat_list(traceout_mat_list): # takes a list of matrices to perform a traceout operation and converts them to a single superoperator
    '''
        traceout_mat_list is a list of matrices provided by partial_trace_matrix_list
    '''
    traceout_mat_rho_vec=None # intially nothing there
    for traceout_mat_psi in traceout_mat_list:
        if type(traceout_mat_rho_vec)==type(None):
            traceout_mat_rho_vec=build_supOP(traceout_mat_psi,traceout_mat_psi.T) # tensored version to add a subspace to vectorized density matrix
        else:
            traceout_mat_rho_vec=traceout_mat_rho_vec+build_supOP(traceout_mat_psi,traceout_mat_psi.T) # otherwise add to existing elements
    return traceout_mat_rho_vec
    
def make_partial_trace_supOp_space_sizes(n_sub,n_before,n_after): # takes the size of the subspaces and builts the traceout superoperator
    '''
        n_sub is the size of the subspace to be traced over
        n_before is the total size of previous subspaces
        n_after is the total size of subsequent subspaces
    '''
    traceout_mat_list=partial_trace_matrix_list(n_sub,n_before,n_after)
    traceout_mat_rho_vec=make_partial_trace_supOp_mat_list(traceout_mat_list)
    return traceout_mat_rho_vec
    
def make_subspace_expand_matrix(initial_subspace_sizes,final_subspace_sizes): # make a matrix to expand given subspaces of a density matrix or state vector
    '''
        initial_subspace_sizes is the initial sizes of all subspaces
        final_subspace_sizes is the final size of all subspaces
        returns a prod(final_subspace_sizes)Xprod(initial_subspace_sizes) sparse matrix which projects into the larger Hilbert space, leaving added modes empty
    '''
    proj_mat=sparse.csr_array(np.ones([1,1])) # initial 1x1 sparse matrix
    for i_kron in range(len(initial_subspace_sizes)): # loop to take tensor products
        kron_mat=sparse.eye_array(initial_subspace_sizes[i_kron]) # start with identity
        kron_mat.resize(final_subspace_sizes[i_kron],initial_subspace_sizes[i_kron]) # add zeros
        proj_mat=sparse.kron(kron_mat,proj_mat) # tensor into the projection matrix
    return sparse.csr_array(proj_mat)
    
def trace_over_subspaces_diag(diag_vec,diag_lengths,trace_subspaces): # traces a diagonal with given lengths in each subspaces for trace_subspaces
    '''
        diag_vec is a vector giving the diagonal
        diag_lengths gives the length of the diagonal in each subspace
        trace_subspaces gives boolean values for whether a subspace should be traced over
    '''
    n_before=1
    for i_trace in range(len(diag_lengths)):
        if trace_subspaces[i_trace]:
            n_after=1
            for i_after in range(i_trace+1,len(diag_lengths)):
                n_after=n_after*diag_lengths[i_after]
            traceout_mat=partial_trace_diag_matrix(diag_lengths[i_trace],n_before,n_after)
            diag_vec=traceout_mat@diag_vec # perform partial trace
        else:
            n_before=n_before*diag_lengths[i_trace]
    return diag_vec
    
    
def subspace_sizes_to_diag_lengths(subspace_sizes,subspace_diags): # convertes the sizes of subspaces to the lengths of diagonals within each subspace
    '''
        subspace_sizes contains the sizes of each subspace
        subspace_diags contains the diagonal index within each subspace
        returns the length of the diagonal within each subspace
    '''
    diag_lengths=[] # empty list
    for i_subspace in range(len(subspace_sizes)):
        diag_lengths=diag_lengths+[subspace_sizes[i_subspace]-abs(subspace_diags[i_subspace])]
    return diag_lengths
    
def truncated_diag_two_photon_loss_tensor(diag_vals,subspace_sizes,subspace_diags,gammat_list=None): # builds a truncated diagonal from two-photon loss
    '''
        diag_vals are the values along a given diagonal
        subspace_sizes are the sizes of the different subspaces (not the length of the diagonals in each)
        subspace_diags are the indices taken within each diagonal, can include negative integers for below the diagonal
        gamma_t list is the amount of two photon loss to apply within each subspace defaults to None, which leads each being assigned a gammat value of 30
        returns the truncated diagonal after two photon loss, sized automatically according to subspace sizes
    '''
    diag_lengths=subspace_sizes_to_diag_lengths(subspace_sizes,subspace_diags)
    if type(gammat_list)==type(None): # if undefined
        gammat_list=[30]*len(subspace_sizes)
    n_before=1
    for i_subspace in range(len(subspace_sizes)):
        n_after=1
        for i_after in range(i_subspace+1,len(subspace_sizes)): # caculate total size of subspace after
            n_after=n_after*diag_lengths[i_after]
        diag_vals=two_photon_loss_diag_lindblad_evolve(diag_vals,diag_index=abs(subspace_diags[i_subspace]),gammat=gammat_list[i_subspace],n_before=n_before,n_after=n_after) # apply lindblad operator to the diagonal
        truncate_array=np.zeros(diag_lengths[i_subspace],dtype=bool)
        if subspace_diags[i_subspace]==0: # if it is a diagonal of the subspace keep first two elements
            truncate_array[0]=True
            truncate_array[1]=True
            
        elif abs(subspace_diags[i_subspace])==1: # if first off diagonal keep first element
            truncate_array[0]=True
        else: # any higher diagonal will have no non-zero values, no point in continuing
            return np.zeros(0) # return an empty array
        truncate_array=np.kron(truncate_array,np.ones(n_before,dtype=bool))
        truncate_array=np.kron(np.ones(n_after,dtype=bool),truncate_array)
        diag_vals=diag_vals[truncate_array] # perform truncation
        if subspace_diags[i_subspace]==0: # if it is a diagonal of the subspace keep first two elements
            n_before=n_before*2 # update size of subspace before
    return diag_vals
    
def diagonal_indices_2photon_loss_truncate(subspace_diags): # determines where the density matrix elements need to be placed in a density matrix after computing a given truncated diagonal
    '''
        subspace_diags contains which diagonals are taken in each subspace, including -1 values for lower diagonals
        returns row and column indices 
    '''
    row_indices=np.zeros(1,dtype=int)
    col_indices=np.zeros(1,dtype=int)
    for i_subspace in range(len(subspace_diags)):
        if subspace_diags[i_subspace]==0: # if this subspace is a main diagonal
            add_indices=np.zeros(2*len(row_indices),dtype=int)
            add_indices[(-len(row_indices)):]=2**i_subspace # second half gets additional values
            row_indices=np.kron(np.ones(2,dtype=int),row_indices)+add_indices # double size and add to second half
            col_indices=np.kron(np.ones(2,dtype=int),col_indices)+add_indices # double size and add to second half
        elif  subspace_diags[i_subspace]==1: # if this subspace is an upper diagonal
            col_indices=col_indices+2**i_subspace # shift column indices while leaving rows the same
        elif  subspace_diags[i_subspace]==-1: # if this subspace is a lower diagonal
            row_indices=row_indices+2**i_subspace # shift row indices while leaving coumns the same
        else:
            return [np.zeros(0),np.zeros(0)] # return empty arrays if any subspace gets a higher diagonal
    return(row_indices,col_indices)

def density_matrix_2photon_loss_from_psis(psi_ket,psi_bra,subspace_sizes,trace_subspaces):  # produces a density matrix (or analog for outer product of off diagonal elements) for state vectors with given subspace sizes, some of which may be traced over, the remainder of which are subject to strong two-photon loss
    '''
        psi_bra is the state vector used as a bra
        psi_ket is the state vector used as a ket
        subspace_sizes are the sizes of each subspace used within the tensor product structure
        trace_subspaces contains booleans which indicate which subspaces should be traced over
    '''
    subspace_sizes=np.array(subspace_sizes,dtype=int)
    n_traced=sum(trace_subspaces) # number of subspaces traced over
    untraced_list=list(np.array(1-np.array(trace_subspaces),dtype=bool))
    subspace_sizes_untraced=subspace_sizes[untraced_list]
    n_untraced=sum(untraced_list)
    subspace_diags_untraced=np.zeros(n_untraced,dtype=int)
    subspace_diags=np.zeros(n_traced+n_untraced,dtype=int)
    rho_final=np.zeros([2**n_untraced,2**n_untraced],dtype=complex) # empty final density matrix
    for i_diags in range(3**n_untraced): # loop over all possible diagonals
        diag_lengths=subspace_sizes_to_diag_lengths(subspace_sizes,subspace_diags)
        diag_vec=compute_diag_psis(psi_ket,psi_bra,subspace_sizes,subspace_diags) # compute the given diagonal
        diag_vec_trace=trace_over_subspaces_diag(diag_vec,diag_lengths,trace_subspaces)
        diag_vec_trunc=truncated_diag_two_photon_loss_tensor(diag_vec_trace,subspace_sizes_untraced,subspace_diags_untraced)
        (row_indices,col_indices)=diagonal_indices_2photon_loss_truncate(subspace_diags_untraced)
        for i_populate in range(len(diag_vec_trunc)):
            rho_final[row_indices[i_populate],col_indices[i_populate]]=copy.copy(diag_vec_trunc[i_populate])
        for i_increment in range(len(subspace_diags_untraced)): # increment the diagonal which is used
            if subspace_diags_untraced[i_increment]==0:
                subspace_diags_untraced[:i_increment]=0
                subspace_diags_untraced[i_increment]=1
                break
            elif subspace_diags_untraced[i_increment]==1:
                subspace_diags_untraced[:i_increment]=0
                subspace_diags_untraced[i_increment]=-1
                break
        subspace_diags[untraced_list]=subspace_diags_untraced
    return rho_final
    

        
def construct_gadget_supOp_two_photon_loss(n_hat_disp_vals,gadg_params={},n_hat_squeeze_vals=[0,0],phi_disp_vals=[0,0],phi_squeeze_vals=[0,0],n_sub=31): # constructs a 16x16 superoperator for the photon loss gadget assuming strong two-photon loss after each pass
    '''
        n_hats_disp is the number of photons of displacement addded to each mode
        gadg_params is the parameters which are passed to the gadget, see QOtrunc.apply_asym_gadget
        n_hats_squeeze is the number of photons of squeezing applied to each mode
        phis_disp is the angles for the displacment on each mode, diffaults to zero on each
        phis_squeeze is the angle for squeezing applied to each mode defaults to zero on each
        outputs a 16x16 superoperator which acts on a vectorized density matrix from the left
        n_sub is the size of the subspace
    '''
    if len(n_hat_disp_vals)>2 or len(n_hat_squeeze_vals)>2 or len(phi_disp_vals)>2 or len(phi_squeeze_vals)>2 or ('n_sub_int' in gadg_params and (not type(gadg_params['n_sub_int'])==type(None))): # if any information has been provided about the interaction mode
        psi_params=[{},{},{}] # list of empty dictionaries
        if 'n_sub_int' in gadg_params and (not type(gadg_params['n_sub_int'])==type(None)): # if a value has been supplied
            psi_params[2]['n_sub']=gadg_params['n_sub_int'] # use supplied value
            output_state_arr=np.zeros([(n_sub**2)*(gadg_params['n_sub_int']**2),4],dtype=complex) # vector for storing the four possible output states
            subspace_sizes=[n_sub,n_sub,gadg_params['n_sub_int'],gadg_params['n_sub_int']]
        else:
            psi_params[2]['n_sub']=n_sub # otherwise default to the same number as the other two modes
            output_state_arr=np.zeros([gadg_params['n_sub']**4,4],dtype=complex)
            subspace_sizes=[n_sub]*4
    else:
        psi_params=[{},{}] # list of empty dictionaries
        output_state_arr=np.zeros([n_sub**4,4],dtype=complex)
        subspace_sizes=[n_sub]*4
    psi_params[0]['n_sub']=n_sub
    psi_params[1]['n_sub']=n_sub
    for i_param in range(len(n_hat_disp_vals)):
        psi_params[i_param]['n_hat_disp']=n_hat_disp_vals[i_param] # displacement photons in each mode
    for i_param in range(len(n_hat_squeeze_vals)):
        psi_params[i_param]['n_hat_squeeze']=n_hat_squeeze_vals[i_param] # squeezing photons in each mode
    for i_param in range(len(phi_disp_vals)):
        psi_params[i_param]['phi_disp']=phi_disp_vals[i_param] # displacement angle for each mode
    for i_param in range(len(phi_squeeze_vals)):
        psi_params[i_param]['phi_squeeze']=phi_squeeze_vals[i_param] # squeezing angle for each mode
    for i_start_0 in range(2): # number of photons in mode 0
        for i_start_1 in range(2): # number of photons in mode 1
            psi_params[0]['start_photons']=i_start_0 # add starting photon information
            psi_params[1]['start_photons']=i_start_1 # add starting photon information
            (psi_tot,psi1,psi2)=QOtrunc.psi_asym_gadget_squeeze_kit(psi_params,gadg_params) # core state-vector calculation for different photon numbers
            output_state_arr[:,i_start_0+2*i_start_1]=copy.copy(psi_tot)
    supOp_final=np.zeros([16,16],dtype=complex)
    trace_subspaces=[False,False,True,True]
    for i_start_0_ket in range(2): # number of photons in mode 0 in the ket
        for i_start_1_ket in range(2): # number of photons in mode 1 in the ket
            for i_start_0_bra in range(2): # number of photons in mode 0 in the bra
                for i_start_1_bra in range(2): # number of photons in mode 1 in the bra
                    input_ind_supOp=i_start_0_bra+2*i_start_1_bra+4*i_start_0_ket+8*i_start_1_ket # use tensor product structure to calculate index
                    psi_ket=copy.copy(output_state_arr[:,i_start_0_ket+2*i_start_1_ket])
                    psi_bra=copy.copy(output_state_arr[:,i_start_0_bra+2*i_start_1_bra])
                    rho_final=density_matrix_2photon_loss_from_psis(psi_ket,psi_bra,subspace_sizes,trace_subspaces) # create final 4x4 density matrix after loss
                    supOp_final[:,input_ind_supOp]=matrix_rho_to_vector_rho(rho_final) # turn to column of density matrix
    return supOp_final
    
def two_mode_loss_n_sub(n_sub_in): # builds a 16x(4**n_in) superoperator which performs strong two-photon loss on two modes with up to (n_sub_in-1) photons in each mode initially  to 0 or 1 photon in each mode
    '''
        n_sub_in is the initial size of each subspace
    '''
    supOp_final=np.zeros([16,n_sub_in**4],dtype=complex)
    trace_subspaces=[False,False]
    subspace_sizes=[n_sub_in,n_sub_in]
    for i_start_0_ket in range(n_sub_in): # number of photons in mode 0 in the ket
        for i_start_1_ket in range(n_sub_in): # number of photons in mode 1 in the ket
            for i_start_0_bra in range(n_sub_in): # number of photons in mode 0 in the bra
                for i_start_1_bra in range(n_sub_in): # number of photons in mode 1 in the bra
                    input_ind_supOp=i_start_0_bra+(n_sub_in)*i_start_1_bra+(n_sub_in**2)*i_start_0_ket+(n_sub_in**3)*i_start_1_ket # use tensor product structure to calculate index
                    psi_ket=np.zeros(n_sub_in**2)
                    psi_ket[i_start_0_ket+n_sub_in*i_start_1_ket]=1
                    psi_bra=np.zeros(n_sub_in**2)
                    psi_bra[i_start_0_bra+n_sub_in*i_start_1_bra]=1
                    rho_final=density_matrix_2photon_loss_from_psis(psi_ket,psi_bra,subspace_sizes,trace_subspaces) # create final 4x4 density matrix after loss
                    supOp_final[:,input_ind_supOp]=matrix_rho_to_vector_rho(rho_final) # turn to column of density matrix
    return supOp_final
    

def two_mode_0_1_HOM(): # constructs a 16x16 matrix which uses the Hong-Ou-Mandel effect along with two-photon loss in each mode to enfoce a maximum of one photon
    bs5050_Op2=linalg_d.expm(QOtrunc.apply_beamsplitting_Op(np.pi/4,9,[3,1],[1,3],gen_mat=True).todense()) # 50:50 beamsplitting operator acting on modes with up to two photons
    bs5050_supOp2=build_supOP(bs5050_Op2,bs5050_Op2.conj().T) # beamsplitting superoperator acting on modes with up to two photons
    bs5050inv_Op1=linalg_d.expm(QOtrunc.apply_beamsplitting_Op(-np.pi/4,4,[2,1],[1,2],gen_mat=True).todense()) # 50:50 beamsplitting operator acting on modes with up to one photon, phases reversed to cancel exactly
    bs5050inv_supOp1=build_supOP(bs5050inv_Op1,bs5050inv_Op1.conj().T) # phase inverted beamsplitting superoperator acting on modes with up to one photon
    mode_expand_Op1_2=make_subspace_expand_matrix([2,2],[3,3]) # operator which expands from one two two photons
    mode_expand_supOp1_2=build_supOP(mode_expand_Op1_2,mode_expand_Op1_2.T) # superoperator for expanding Hilbert spaces
    supOp_HOM=bs5050inv_supOp1@two_mode_loss_n_sub(3)@bs5050_supOp2@mode_expand_supOp1_2 # chain Hilbert space expansion, beamsplitting, two photon loss on each mode, and beamsplitting again together for total operation
    return supOp_HOM
    
def calculate_two_mode_covariance(rho,n_before,n_after): # builds an operator matrix to calculate the covaraince between two modes
    '''
    rho is a density matrix in either vector or array form
    n_before and n_after are lists of the number before and after for each mode such that n_before[0] and n_after[0] correspond to the first mode and n_before[1] and n_after[1] correspond to the second mode
    returns operator to calculate the covariance between two modes
    '''
    rho_m=convert_rho_to_matrix_idem(rho) # convert to matrix
    total_size=rho_m.shape[0] # size of matrix
    n_sub_0=QOtrunc.get_subspace_photons(total_size,n_before[0],n_after[0]) # size of first subspace
    a_sub_0=QOtrunc.make_annihiltion_Op(n_trunc=n_sub_0)
    a_0=QOtrunc.tensor_identities(a_sub_0,n_before[0],n_after[0]) # creation operator for mode 1
    n_0=a_0.conj().T@a_0 # number operator
    n_sub_1=QOtrunc.get_subspace_photons(total_size,n_before[1],n_after[1]) # size of second subspace
    a_sub_1=QOtrunc.make_annihiltion_Op(n_trunc=n_sub_1)
    a_1=QOtrunc.tensor_identities(a_sub_1,n_before[1],n_after[1]) # creation operator for mode 1
    n_1=a_1.conj().T@a_1 # number operator
    cov=calculate_Op_expectation(rho,n_0@n_1)-calculate_Op_expectation(rho,n_0)*calculate_Op_expectation(rho,n_1)
    return cov

def maximally_mixed_trace_test(supOp): # checks the trace of a maximally mixed state after a superoperator is applied, should be 1 up to numerical error for a valid superoperator
    '''
        supOp is a superoperator
    '''
    dm_size=int(np.round(np.sqrt(supOp.shape[1]))) # square root of the dimension which will be multipled
    mm_rho_vec=matrix_rho_to_vector_rho(np.eye(dm_size)/dm_size) # appropriate sized maximally mixed state
    mm_rho_vec.shape
    rho_f_vec=supOp@mm_rho_vec # apply superoperator
    trace_fin=sum(np.diag(vectorized_rho_to_matrix_rho(rho_f_vec))) # take trace
    return trace_fin
    
def construct_nl_expand_trace_gen_supOp(gamma=np.pi/(2*np.sqrt(2)),n_trunc=2,eta=0): # constructs a generator for a nonlinear operation along with matrices to expand the basis and perform tracing
    '''
        gamma is the total amount which the nonlinear operation is applied, defaults to np.pi/(2*np.sqrt(2)) which corresponds to full transfer to the pump mode
        n_trunc is the size of the signal mode, from which the maximum size of the pump mode is inferred, defaults to 2, which is the minumum size to see non-trivial effects
        eta is the rate of single-photon loss from the pump mode, defailts to 0
        returns supOp_nl which is the superoperator for performing the non-linear operations, expand_mat_rho_vec which is a matrix which adds the pump mode to the hilbert space, and traceout_mat_rho_vec_list which is a list of matrices which can be used to perform tracing out
        
    '''
    n_max_pump=int(np.floor(n_trunc/2)) # maximum possible number of photons in the pump mode
    expand_mat_psi=make_subspace_expand_matrix([n_trunc+1,1],[n_trunc+1,n_max_pump+1]) # add subspace with vaccuum to wavefunction
    expand_mat_rho_vec=build_supOP(expand_mat_psi,expand_mat_psi.T) # tensored version to add a subspace to vectorized density matrix
    traceout_mat_psi_list=partial_trace_matrix_list(n_max_pump+1,n_before=n_trunc+1,n_after=1) # matrix to trace out just one subspace
    traceout_mat_rho_vec=None # intially nothing there
    for traceout_mat_psi in traceout_mat_psi_list:
        if type(traceout_mat_rho_vec)==type(None):
            traceout_mat_rho_vec=build_supOP(traceout_mat_psi,traceout_mat_psi.T) # tensored version to add a subspace to vectorized density matrix
        else:
            traceout_mat_rho_vec=traceout_mat_rho_vec+build_supOP(traceout_mat_psi,traceout_mat_psi.T) # otherwise add to existing elements
    H_nl=QOtrunc.apply_chi_2_deg_Op(gamma,(n_trunc+1)*(n_max_pump+1),n_before=[1,n_trunc+1],n_after=[n_max_pump+1,1],gen_mat=True) # make nonlinear Hamiltonian
    supOp_nl=build_coherent_supOp(1j*H_nl) # build nonlinear terms
    if not eta==0: # if the pump mode has non-zero loss
        a_pump=QOtrunc.make_annihiltion_Op(n_trunc=n_max_pump) # anihilation operator
        a_pump=QOtrunc.tensor_identities(a_pump,n_before=n_trunc+1,n_after=1) # tensor identities before
        pump_loss_supOp=build_lindblad_supOp(a_pump) # make a superoperator for loss on the pump mode
        supOp_nl=supOp_nl+eta*pump_loss_supOp # add to overall superoperator
    return (supOp_nl,expand_mat_rho_vec,traceout_mat_rho_vec)
    
    
def Zeno_blockade_displacement_supOp(gamma_loss,alpha_inj,n_trunc=5,is_coherent=False,eta_pump_loss=0,return_gen=False): # two photon loss and displacement to implement a Zeno blockade based flip between zero and one photon
    '''
        gamma_loss is the strength of the two-photon loss
        alpha_inj is the strength of the photon injection
        returns a superoperator in the 0-1 subspace of the initial mode
        n_trunc is the maximum number of photons allowed when creating the superoperator
        is_coherent is a boolean switch which determines whether the Zeno effect is performed in a coherent or incoherent way, in other words whether the photon in the pump mode leaves the system or remains
        eta_pump_loss is the loss rate for the pump mode (only relevant when is_coherent==True, defaults to 0
        return_gen is a boolean flag which determines whether the generator is returned
    '''
    if not is_coherent: # incoherent mode of operartion
        a=QOtrunc.make_annihiltion_Op(n_trunc=n_trunc) # annihilation operator
        a_d=a.conj().T # Hermitian conjugate
        supOp_2photon_loss=build_lindblad_supOp(a@a) # build two-photon loss superoperator
        H_disp=QOtrunc.apply_displacement_Op(alpha_inj,n_trunc+1,n_before=1,n_after=1,gen_mat=True) # make displacement Hamiltonian
        supOp_displace=build_coherent_supOp(1j*H_disp) # build coherent terms
        if not return_gen:
            evo_supOp=linalg_d.expm(supOp_displace.todense()+gamma_loss*supOp_2photon_loss.todense()) # exponentiate to build time evolution terms
        else:
            evo_supOp=supOp_displace.todense()+gamma_loss*supOp_2photon_loss.todense() # no exponentiation
    else: # coherent operation requires adding a subspace, evolving coherently and tracing out
        n_max_pump=int(np.floor(n_trunc/2)) # maximum possible number of photons in the pump mode
        (supOp_nl,expand_mat_rho_vec,traceout_mat_rho_vec)=construct_nl_expand_trace_gen_supOp(gamma_loss,n_trunc=n_trunc,eta=eta_pump_loss) # construct non-linear superoperator  (with any pump-mode loss) and tracing and expanding operations
        H_disp=QOtrunc.apply_displacement_Op(alpha_inj,(n_trunc+1)*(n_max_pump+1),n_before=1,n_after=n_max_pump+1,gen_mat=True) # make displacement Hamiltonian
        supOp_displace=build_coherent_supOp(1j*H_disp) # build coherent terms
        if not return_gen:
            evo_supOp=linalg_d.expm(supOp_displace.todense()+supOp_nl.todense()) # exponentiate to build time evolution terms
            evo_supOp=traceout_mat_rho_vec@evo_supOp@expand_mat_rho_vec # full superoperator including expanding and tracing
        else:
            return (supOp_displace.todense()+supOp_nl.todense(),traceout_mat_rho_vec,expand_mat_rho_vec) # return generators along with expansion and traceout
    return evo_supOp
    

def expand_nl_phase_nl_trace_supOP(gamma=np.pi/(2*np.sqrt(2)),phase_apply=0,n_trunc=2): # builds a superoperator which expands the Hilbert space to add an empty pump mode, applies non-linearity, applies a phase, re-applies non-linearity, and then traces the pump mode out
    '''
        gamma is the strenght of the nonlinearity applied each time, defaults to np.pi/(2*np.sqrt(2)) which corresponds to full transfer to the pump mode
        phase_apply is the intermediate phase which is applied to the pump mode, defualts to 0
        n_trunc is the size of the signal mode, from which the maximum size of the pump mode is inferred, defaults to 2, which is the minumum size to see non-trivial effects
    '''
    n_max_pump=int(np.floor(n_trunc/2)) # maximum possible number of photons in the pump mode
    (supOp_nl,expand_mat_rho_vec,traceout_mat_rho_vec)=construct_nl_expand_trace_gen_supOp(gamma,n_trunc=n_trunc) # construct non-linear superoperator and tracing and expanding operations
    supOp_apply_nl=linalg_d.expm(supOp_nl.todense()) # superoperator to apply non-linearity
    H_phase=QOtrunc.apply_time_evolution_Op(phase_apply,(n_trunc+1)*(n_max_pump+1),n_before=n_trunc+1,n_after=1,gen_mat=True) # make displacement Hamiltonian
    supOp_apply_phase=linalg_d.expm(build_coherent_supOp(1j*H_phase).todense()) # superoperator which applies a phase
    full_supOp=traceout_mat_rho_vec@supOp_apply_nl@supOp_apply_phase@supOp_apply_nl@expand_mat_rho_vec # string all operations together
    return full_supOp

'''
def compute_tensor_expansion_matrix_supOp(rho_size,n_before,n_after): # pre-computes a matrix which acts as a tensor product with identities over other modes
    '
        rho_size is the linear size of a density matrix (superoperator acting on vectorized version is therefore of size rho_size**2)
        n_before is the size of the subspace before
        n_after is the size of the subspace after
    '
    expand_mat_i_list=[] # empty list for first indices
    expand_mat_j_list=[] # empty list for second indices
    for i_rho in range(rho_size**2): # loop through different elements
        rho_v=np.zeros(rho_size**2)
        rho_v[i_rho]=1
        rho_m=sparse.csr_array(vectorized_rho_to_matrix_rho(rho_v)) # convert to a sparse matrix
        rho_kron_m=QOtrunc.tensor_identities(rho_m,n_before,n_after) # tensor identity matrices in
        rho_kron_v=matrix_rho_to_vector_rho(rho_kron_m) # convert back to vector
        one_inds=rho_kron_v.nonzero()[1] # find non-zero indices
        expand_mat_i_list=expand_mat_i_list+list(i_rho*np.ones(len(one_inds),dtype=int)) # add to first index list
        expand_mat_j_list=expand_mat_j_list+list(one_inds) # add to second index list
    expand_mat=sparse.csr_array((np.ones(len(expand_mat_i_list),dtype=int),(expand_mat_i_list,expand_mat_j_list))) # create sparse matrix for expansion
    return expand_mat
'''

def compute_tensor_expansion_matrices_supOp(rho_size,n_before,n_after): # pre-computes a matrix which acts as a tensor product with identities over other modes
    '''
        rho_size is the linear size of a density matrix (superoperator acting on vectorized version is therefore of size rho_size**2)
        n_before is the size of the subspace before
        n_after is the size of the subspace after
    '''
    expand_mat_list=[] # empty list to fill with projector matrices
    total_size=(n_before**2)*(rho_size**2)*(n_after**2)
    for iBefore in range(n_before):
        for jBefore in range(n_before):
            before_mat=sparse.csr_array(([1],([iBefore],[jBefore])),shape=(n_before,n_before))
            for iAfter in range(n_after):
                for jAfter in range(n_after):
                    after_mat=sparse.csr_array(([1],([iAfter],[jAfter])),shape=(n_after,n_after))
                    expand_mat_i_list=[] # empty list for first indices
                    expand_mat_j_list=[] # empty list for second indices
                    for i_rho in range(rho_size**2): # loop through different elements
                        rho_v=np.zeros(rho_size**2)
                        rho_v[i_rho]=1
                        rho_m=sparse.csr_array(vectorized_rho_to_matrix_rho(rho_v)) # convert to a sparse matrix
                        #rho_kron_m=QOtrunc.tensor_identities(rho_m,n_before,n_after) # tensor identity matrices in
                        rho_kron_m=sparse.kron(after_mat,sparse.kron(rho_m,before_mat)) # tensor the other indices in
                        rho_kron_v=matrix_rho_to_vector_rho(rho_kron_m) # convert back to vector
                        one_inds=rho_kron_v.nonzero()[1] # find non-zero indices
                        expand_mat_i_list=expand_mat_i_list+[i_rho] # add to first index list
                        expand_mat_j_list=expand_mat_j_list+list(one_inds) # add to second index list
                    expand_mat=sparse.csr_array((np.ones(len(expand_mat_i_list),dtype=int),(expand_mat_i_list,expand_mat_j_list)),shape=(rho_size**2,total_size)) # create sparse matrix for expansion
                    expand_mat_list=expand_mat_list+[expand_mat] # add to list which can be used to construct tensor product
    return expand_mat_list

def build_tensor_product_from_expand_mat_list(mat,expand_mat_list): # uses a list of pre-computed projector matrices to perform tensor products more efficiently
    '''
        mat is the matrix to be expanded to a larger subspace
        expand_mat_list is a list of projectors to expand into the larger space
    '''
    for i_expand_mat in range(len(expand_mat_list)): # loop to build the overall matrix
        if i_expand_mat==0: # special case for first one
            expanded_mat=expand_mat_list[0].T@mat@expand_mat_list[0]
        else:
            expanded_mat=expanded_mat+expand_mat_list[i_expand_mat].T@mat@expand_mat_list[i_expand_mat]
    return expanded_mat
    
    
    
def construct_expm_binary_approx(gen_mat,max_exp=np.pi,num_powers=30): # construct matrix exponentiation in powers of 2 to avoid expensive inner-loop matrix exponentiation
    '''
        gen_mat is the matrix to be exponentiated
        max_exp is the largest multiple in front of the generator
        num_powers is the number of powers to apply
        returns a dictionary containing the different exponentiation strengths and powers
    '''
    gen_exp_list=[] # list of exponentiated matrices
    multiple_list=[] # list of multiples in the exponent
    for i_power in range(num_powers): # loop through different powers
        multiple=max_exp*2**(-i_power) # find the number to multiply the generator by
        multiple_list=multiple_list+[copy.copy(multiple)] # add to list
        gen_exp_list=gen_exp_list+[linalg_d.expm(multiple*gen_mat)] # perform matrix exponentiation and add to list
    fast_expm_dict={'multiple_list':multiple_list,'gen_exp_list':gen_exp_list}
    return fast_expm_dict
    
def fast_expm_from_dict(fast_expm_dict,multiple): # uses a binary tree to use precalculation to speed up matrix exponentiation
    '''
        fast_expm_dict is a pre-computed dictionary to calculate the matrix exponential
        multiple is the factor which is multiplied when exponentiation is performed
    '''
    remainder=copy.copy(multiple) # the remaining part of the multiple
    total_expm=np.eye(fast_expm_dict['gen_exp_list'][0].shape[0]) # start with identity matrix
    if remainder>(2*fast_expm_dict['multiple_list'][0]):
        multiply_pow=int(np.floor(remainder/fast_expm_dict['multiple_list'][0]))
        remainder=remainder-fast_expm_dict['multiple_list'][0]*multiply_pow # subtract from remainder
        for i_pow in range(multiply_pow):
            total_expm=total_expm@fast_expm_dict['gen_exp_list'][0]
    for i_multiple in range(len(fast_expm_dict['multiple_list'])): # checks different multiples
        if remainder>fast_expm_dict['multiple_list'][i_multiple]: # if there is enough rotation to include the next power
            total_expm=total_expm@fast_expm_dict['gen_exp_list'][i_multiple] # multiply by matrix from list
            remainder=remainder-fast_expm_dict['multiple_list'][i_multiple] # reduce remainder
    return total_expm
            
