import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalgs
import scipy.linalg as linalg
import copy as copy


def kron_ith_position(M,i_pos,n_var): # uses tensor products to create a matrix with an identity at all positions except for the ith assumes all variables have the same size which is the size of M
    '''
        M is the matrix to be place at position i_pos
        i_pos is the position where the matrix should be placed
        n_var is the total number of variables
        returns the matrix as a sparse csr matrix
    '''
    var_size=len(M)   # the size of each variable
    prepend_identity=sparse.identity(var_size**i_pos,format='csr') # indentity to prepend as a sparse matrix
    append_identity=sparse.identity(var_size**(n_var-i_pos-1),format='csr') # indentity to append as a sparse matrix
    kron_mat=sparse.csr_array(sparse.kron(sparse.kron(prepend_identity,sparse.csr_array(M)),append_identity)) # perform tensor product
    return kron_mat
    
def construct_quadratic_Hamiltonian(M,J,h): # constructs a quadratic Hamiltonian built from matrix M
    '''
        M is the matrix used to construct linear and quadratic terms
        J is an array of quadratic terms, the upper diagonal of which is used to construct the Hamiltonian matrix
        h is a vector of linear terms
        returns the matrix as a sparse csr matrix
    '''
    n_var=len(h) # number of terms in Hamiltonian
    H=sparse.csr_array((len(M)**n_var,len(M)**n_var)) # empty sparse matrix
    for i_add in range(n_var): # loop through first index
        H=H+h[i_add]*kron_ith_position(M,i_add,n_var) # add linear term
        for j_add in range(i_add,n_var): # loop through second index
            if not J[i_add,j_add]==0: # don't perform tensor products for terms which don't contribute anyway
                H=H+J[i_add,j_add]*kron_ith_position(M,i_add,n_var)@kron_ith_position(M,j_add,n_var) # dot prodoct of tensor products
    return H
    
def kron_ith_position_vec(V,i_pos,n_var): # uses tensor products to create a matrix with an identity at all positions except for the ith assumes all variables have the same size which is the size of M
    '''
        M is the vector to be place at position i_pos
        i_pos is the position where the vector should be placed
        n_var is the total number of variables
        returns a 1D vector
    '''
    var_size=len(V)   # the size of each variable
    prepend_identity=np.ones(var_size**i_pos) # indentity to prepend as a sparse matrix
    append_identity=np.ones(var_size**(n_var-i_pos-1)) # indentity to append as a sparse matrix
    kron_vec=np.kron(np.kron(prepend_identity,V),append_identity) # perform tensor product
    return kron_vec


def rotate_disp_ham(theta): # pure imaginary Hamiltonian used to rotate the basis out of undefined state for a single binary variable
    '''
         theta is an angle used to rotate between an undefined state and the |+> state, theta=0 defines full rotation into the undefined state, theta= pi/2 is fully in |+>
         returns a 3x3 purely imaginary matrix
    '''
    decay_state=np.array([-np.sin(theta),np.cos(theta)/np.sqrt(2),np.cos(theta)/np.sqrt(2)]) # vector representing decaying state
    H=-1j*np.outer(decay_state,np.conj(decay_state))  # purely imaginary term made from outer product
    return H


def multi_qubit_rotate_disp_ham(theta,n_qubit): # creates a multi-qubit dissipation hamiltonian using tensor products
    '''
        theta is an angle used to rotate between an undefined state and the |+> state, theta=0 defines full rotation into the undefined state, theta= pi/2 is fully in |+>
        n is the number of qubit variables, n.b. because of the undefined state, the size of the total matrix is 3^n_qubit
    '''
    H_single=rotate_disp_ham(theta) # imaginary Hamiltonian for a single angle
    H_tot=sparse.csr_array((3**n_qubit,3**n_qubit)) # empty array
    for i_pos in range(n_qubit): # empty starting matrix
        H_tot=H_tot+kron_ith_position(H_single,i_pos,n_qubit) # add new matrix
    return H_tot
    
def multi_qubit_measure_project_xi(theta,n_qubit): # creates a projector which projects out the xi state at each qubit
    '''
        theta is an angle used to rotate between an undefined state and the |+> state, theta=0 defines full rotation into the undefined state, theta= pi/2 is fully in |+>
        n is the number of qubit variables, n.b. because of the undefined state, the size of the total matrix is 3^n_qubit
    '''
    P_single=np.eye(3)-1j*rotate_disp_ham(theta) # projector for a single qubit
    P=np.eye(3**n_qubit) # identity
    for i_pos in range(n_qubit): # empty starting matrix
        P=np.dot(P,kron_ith_position(P_single,i_pos,n_qubit).todense()) # multiply in new matrix note that this is a dense matrix in the computational basis for most values of theta
    return P
    
    
def multi_var_u_offset_ham(n_qubit): # creates a unit offset for the |u> state on n_qubit variables
    '''
        n_qubit is the number of variables on which to create the offset
    '''
    H_single=np.zeros([3,3]) # three by three
    H_single[0,0]=1 # 1 on first element
    H_tot=sparse.csr_array((3**n_qubit,3**n_qubit)) # empty array
    for i_pos in range(n_qubit): # empty starting matrix
        H_tot=H_tot+kron_ith_position(H_single,i_pos,n_qubit) # add new matrix
    return H_tot
    
def sat_var_three_diag(var_list,neg_list,n_qubit): # creates sat clauses in the size 3 variable formulation, diagonal of purely imaginary Hamiltonian
    '''
        var_list is the list of variables involved in the clause
        neg_list is one if the variable is negated and zero otherwise
        n_qubit is the number of variables used
    '''
    H_tot_diag=-1j*np.ones(3**n_qubit) # start with identity matrix
    for i_var in range(len(var_list)): # loop through variables
        if neg_list[i_var]: # if variable is negated
            M=np.array([0,0,1]) # clause can only not be satistisfied if variable is 1
        else:
            M=np.array([0,1,0]) # clause can only not be satistisfied if variable is 0
        H_tot_diag=H_tot_diag*kron_ith_position_vec(M,var_list[i_var],n_qubit) # multiply new term into Hamiltonian
    return H_tot_diag
    
def sat_var_two_diag(var_list,neg_list,n_qubit): # creates sat clauses in the binary variable formulation, diagonal of purely imaginary Hamiltonian
    '''
        var_list is the list of variables involved in the clause
        neg_list is one if the variable is negated and zero otherwise
        n_qubit is the number of variables used
    '''
    H_tot_diag=-1j*np.ones(2**n_qubit) # start with identity matrix
    for i_var in range(len(var_list)): # loop through variables
        if neg_list[i_var]: # if variable is negated
            M=np.array([0,1]) # clause can only not be satistisfied if variable is 1
        else:
            M=np.array([1,0]) # clause can only not be satistisfied if variable is 0
        H_tot_diag=H_tot_diag*kron_ith_position_vec(M,var_list[i_var],n_qubit) # multiply new term into Hamiltonian
    return H_tot_diag

    
def build_sat_Ham_three_diag(var_lists,neg_lists,n_qubit): # adds diagonals to build a purely imaginary satisfiability Hamiltonian
    '''
        var_lists is a list of lists of variables in each clause
        neg_lists is a list of lists of which terms in each clause is negated
        n_qubit is the number of qubits used for the Hamiltonian
        returns the diagonal of the purely imaginary Hamiltonian
    '''
    H_tot_diag=np.zeros(3**n_qubit) # start with no clauses
    for i_clause in range(len(var_lists)): # loop to add clauses
        H_tot_diag=H_tot_diag+sat_var_three_diag(var_lists[i_clause],neg_lists[i_clause],n_qubit) # add to the diagonal
    return H_tot_diag
    


def create_random_unique0_3sat_three(n_qubit): # adds random three satisfiability clauses with at least one literal negated until all zeros is the only satisfying assignment
    '''
        n_qubit is the size of clause
        returns var_lists and neg_lists, a list of variables involved in each clause and which are negated
    '''
    H_tot_diag_two=np.zeros(2**n_qubit) # start with no clauses
    var_lists=[] # list of lists of variables
    neg_lists=[] # list of lists of negations
    while sum(H_tot_diag_two==0)>1: # while more than just the all zeros state is forbidden
        var_list=list(np.random.choice(list(range(n_qubit)),3,replace=False)) # list of variables
        neg_list=[None]*3 # empty list
        neg_list[0]=True # always negate first variable
        neg_list[1]=bool(np.random.randint(0,2)) # randomly negate second variable
        neg_list[2]=bool(np.random.randint(0,2)) # randomly negate third variable
        compare_check=[var_lists[i_list]==var_list and neg_lists[i_list]==neg_list for i_list in range(len(var_lists))] # use list comprehension to list any places both variables and negations match previous entries
        if any(compare_check): # if clause exactly matches one already present
            continue # generate new clause
        var_lists=var_lists+[var_list] # add to list of variables
        neg_lists=neg_lists+[neg_list] # add to list of negations
        H_tot_diag_two=H_tot_diag_two+sat_var_two_diag(var_list,neg_list,n_qubit) # add to diagonal
    return (var_lists,neg_lists)

def sat_var_three(var_list,neg_list,n_qubit): # creates sat clauses in the size 3 variable formulation, purely imaginary Hamiltonian
    '''
    var_list is the list of variables involved in the clause
    neg_list is one if the variable is negated and zero otherwise
    n_qubit is the number of variables used
    '''
    H_tot=-1j*sparse.identity(3**n_qubit,format='csr') # start with idenity matrix
    for i_var in range(len(var_list)): # loop through variables
        if neg_list[i_var]: # if variable is negated
            M=np.diag([0,0,1]) # clause can only not be satistisfied if variable is 1
        else:
            M=np.diag([0,1,0]) # clause can only not be satistisfied if variable is 0
        H_tot=H_tot*kron_ith_position(M,var_list[i_var],n_qubit) # multiply new term into Hamiltonian
    return H_tot
    

    
def domain_wall_sat_var_three(n_qubit,qubit_list=None): # builds satisfiability clauses which enforce a domain-wall constraint
    '''
        n_qubit is the total number of binary variables
        vars is a list of variables (in order) which are to be used in a domain wall encoding, if left as None assumes that all variables are involved in the encoding in order
        returns the matrix which encodes the contraint
    '''
    if type(qubit_list)==type(None): # include all in order if not specified
        qubit_list=list(range(n_qubit))
    H_diag=np.zeros(3**n_qubit) # empty diagonal
    for iClause in range(len(qubit_list)-1): # create necissary clauses
        H_diag=H_diag+sat_var_three_diag([qubit_list[iClause],qubit_list[iClause+1]],[False,True],n_qubit) # forbid anti-domain-wall
    H=sparse.csr_array((H_diag,(list(range(len(H_diag))),list(range(len(H_diag))))),shape=(len(H_diag),len(H_diag))) # convert to diagonal sparse array
    return H
    
    
def expm_multiply_eigh(H,time_prefactor,state_vec,E_V_precomp=None): # function which performs matrix exponentiation of a Hermitian matrix and multiples by a state vector using matrix diagonalisation
    '''
        H is a Hermitian matrix which is diagonalised using eigh, should be in a dense format, not a sparse one
        time_prefactor is a constant which is multiplied in before exponentiating the diagonal elements, note that a real prefactor corresponds to imaginary time (dissipation) while an imaginary number corresponds to time evolution
        state_vec is at state vector to be multiplied by the exponent
        E_V_precomp is a set of precomputed engenvalues and eigenvectors, which can be used to save computational effort
    '''
    if type(E_V_precomp)==type(None): # if no precomputed E and V are provided then we need to perform matrix diagonalisation
        if not linalg.ishermitian(H): # check if matrix is Hermitian
            print('expm_multiply_eigh requires a Hermitian matrix, a non-Hermitian matrix was supplied, returning a NaN value this is likely to cause errors downstream')
        return np.nan
        [E,V]=linalg.eigh(H) # diagonalise the matrix
    else: # if precomputed values have been provided, use those
        [E,V]=E_V_precomp
    state_vec_diag=np.dot(V.conj().T,state_vec) # transform into the diagonal basis
    diag_exp=np.exp(time_prefactor*E) # exponentiate matrix diagonal
    state_vec=np.dot(V,diag_exp*state_vec_diag) # multiply by exponentiated diagonal and return to orginal basis
    return state_vec
    
def precompute_E_V_eigh(H_s,num_step): # precompute the eigenvectors and eignevalues to do an matrix diagonalisation based sweep
    '''
        H_s is a function which returns the Hamiltonian as function of the annealing parameter 0<s<1
        num_step is the number of steps
    '''
    E_V_precomp_list=[None]*num_step
    for i_step in range(num_step): # steps of the evolution
        s=i_step/(num_step-1) # s value
        if linalg.ishermitian(H_s(s).todense()): # real time evolution
            [E,V]=linalg.eigh(H_s(s).todense())
        elif linalg.ishermitian(-1j*H_s(s).todense()): # imaginary time evolution
            [E,V]=linalg.eigh(-1j*H_s(s).todense())
        else:
            print('precompute_E_V_eigh called with matrix is not hermitian or anti- hermitian returning nan value, this is likely to cause errors downstream')
            return np.nan
        E_V_precomp_list[i_step]=[E,V] # add values to list
    return E_V_precomp_list # return list

def zeno_sweep_H_expm_mult(H_s,t_tot,start_state,measure_func,num_step=1000,method='sparse_linalg',E_V_precomp_list=None): # function for performing a generalised anneal with a complex Hamiltonian
    '''
        H_s is a function which returns the Hamiltonian as a function of the annealing parameter 0<s<1
        t_tot is the total time for the entire sweep
        start_state is the initial state of the system
        measure_func is a function which returns the measures used to make plots to avoid returning full (potentially very large) vectors at each step
        num_step is the number of steps used for the simulation, (defaults to 1000)
        E_V_precomp_list is an optional list of precomputed eigenvectors and eigenvalues which save computational effort when doing multiple runs at different runtimes, only relevant if "eigh" method is used
    '''
    state_vec=start_state # initialise system
    if not type(E_V_precomp_list)== type(None):
        num_step=len(E_V_precomp_list) # overwrite number of steps with length of list
    data_list=[None]*num_step # list of outputs of measure func
    for i_step in range(num_step): # steps of the evolution
        s=i_step/(num_step-1) # s value
        if method == 'sparse_linalg': # if we are using the built-in function
            state_vec=sparse.linalg.expm_multiply(-1j*(t_tot/num_step)*H_s(s),state_vec) # matrix exponetiation and multiplication
        elif method == 'eigh': # use custom method based on matrix diagonalisation (better for long times)
            #print(i_step)
            if type(E_V_precomp_list)== type(None): # if no precomputed values were provided
                E_V_precomp=None # pass a None value whic will be ignored
            else:
                E_V_precomp=E_V_precomp_list[i_step] # take precomputed values
            if linalg.ishermitian(H_s(s).todense()): # real time evolution
                state_vec=expm_multiply_eigh(H_s(s).todense(),-1j*(t_tot/num_step),state_vec,E_V_precomp=E_V_precomp) # matrix exponetiation and multiplication
            else: # imaginary time evolution
                state_vec=expm_multiply_eigh(-1j*H_s(s).todense(),(t_tot/num_step),state_vec,E_V_precomp=E_V_precomp) # matrix exponetiation and multiplication
        else: # otherwise method is not recognised and should return nan
            print('zeno_sweep_H_expm_mult received an unrecognised method '+method+' returning a NaN value this is likely to cause errors downstream')
            return np.nan
            
        data_list[i_step]=measure_func(state_vec) # measure desired quantities
    return data_list

def qubit_variable_pm_tilde_transform(theta): # Gives a single variable transform to the plus tilde minus tilde basis
    '''
        theta is the rotation angle for the projection in the three-state space
        P is a matrix where the first column corresponds to + tilde and the second to - tilde in the 0, 1 basis
    '''
    P=np.array([[np.cos(theta),1],[np.cos(theta),-1]])/np.sqrt(2)
    return P

def project_to_allowed_states(allowed_state): # allowed states is a boolean vector of states which are allowed
    '''
        allowed_states is vector with zeros corresponding to non-allowed states
        P is a matrix which projects from a larger set of states to a smaller set of allowed states
    '''
    nz_positions=list(np.where(allowed_state)[0]) # find non-zero elements and convert to list
    indices=[(nz_positions[i],i) for i in range(len(nz_positions))] # use list comprehension to list coordinates
    P=sparse.csr_array(np.ones(len(nz_positions)),indices,shape=[len(allowed_states),len(nz_positions)]) # generate sparse matrix
    return P
    
def list_Hamming_weight(n): # lists the Hamming weights of all computational basis states from 0 to 2^(n-1)
    '''
        n is the number of bits used
    '''
    Hamming_weights=np.zeros(2**n) # array to store Hamming weights
    bin_vec=np.zeros(n) # start with array of all zeros
    for i_weight in range(2**n): # loop through Hamming weights
        Hamming_weights[i_weight]=sum(bin_vec) # current Hamming weight
        for i_incr in range(n): # increment binary vector
            if bin_vec[i_incr]==0:
                bin_vec[:i_incr]=0 # reset previous values to 0
                bin_vec[i_incr]=1 # set bit to 1
                break # break out of incrementation for loop
    return Hamming_weights # return weights
    
def count_projector_mat(n,projector_test): # builds a projector which projects to states where Hamming weights pass the projector test
    '''
        n is the number of bits used
        projector_test is a function which takes a nunber and returns a boolean value to test if a state with a given Hamming weight will be included in the projector
    '''
    Hamming_weights=list_Hamming_weight(n) # list Hamming weights
    indices=[i_Ham for i_Ham in range(len(Hamming_weights)) if projector_test(Hamming_weights[i_Ham])] # list comprehension to produce indices
    P=sparse.csr_array((np.ones(len(indices)),(indices,indices)),shape=(len(Hamming_weights),len(Hamming_weights))) # generate sparse matrix
    return P
    
def list_weights_ranges(n): # lists the ranges of Hamming weights which are possible with u states taking different zero versus 1 values
    '''
        n is the number of |u>,|0>,|1> variables considered
        returns a list of arrays containing the range of allowed values
    '''
    lower_weights=np.zeros(3**n) # lowest Hamming weight obtained when |u> are treated as 0
    for i_sum_lower in range(n): # loop through positions to sum
        lower_weights=lower_weights+kron_ith_position_vec([0,0,1],i_sum_lower,n) # add to weights
    upper_weights=np.zeros(3**n) # lowest Hamming weight obtained when |u> are treated as 1
    for i_sum_upper in range(n): # loop through positions to sum
        upper_weights=upper_weights+kron_ith_position_vec([1,0,1],i_sum_upper,n) # add to weights
    weight_range_list=[None]*(3**n)# list of ranges
    for i_range in range(3**n):
        weight_range_list[i_range]=np.array(range(int(lower_weights[i_range]),int(upper_weights[i_range]+1)))
    return weight_range_list
    
def project_forbidden_weights(n,projector_test,qubit=False): # projects out any state where projector_test cannot be followed regardless of how different |u> values are treated
    '''
        n is the number of |u>,|0>,|1> variables considered
        projector_test is the test values must pass to not be projected out
        the qubit flag creates the same projector, but for binary variables, without the |u> state
    '''
    if not qubit:
        weight_range_list=list_weights_ranges(n) # list of ranges of Hamming weights
    else:
        weights=lower_weights=np.zeros(2**n)
        for i_sum in range(n): # loop to create weights
            weights=weights+kron_ith_position_vec([0,1],i_sum,n) # add tensor product
        weight_range_list=[np.array([weights[i_weight]]) for i_weight in range(2**n)] # use list comprehension to create list
    indices=[i_Ham for i_Ham in range(len(weight_range_list)) if not any(projector_test(weight_range_list[i_Ham]))]
    P=sparse.csr_array((np.ones(len(indices)),(indices,indices)),shape=(len(weight_range_list),len(weight_range_list))) # generate sparse matrix
    return P

    

def project_H_allowed_state_space(H,thresh=10**-9,k=50): # projects to the zero eigenvalue subspace of a Hermitian matrix H
    '''
    returns vectors which project to the zero eigenspace of Hermitian (either sparse or dense) matrix H
    thresh is the threshold below which an eigenvalue is considered zero
    k is the number of eigenvalues to produce when diagonalising (ignorned if H is a dense matrix)
    '''
    if not sparse.issparse(H): # if it is a dense matrix
        (E,V)=linalg.eigh(H) # diagonalise the matrix
        allowed_states=np.where(abs(E)<thresh)[0]
    else:
        (E,V)=linalgs.eigsh(H,k=k,which='SM') # diagonalise the matrix
        allowed_states=np.where(abs(E)<thresh)[0]
        while len(allowed_states==k): # if only zero eigenvalues were found
            k=2*k # double k and try again
            (E,V)=linalgs.eigsh(H,k=k,which='SM') # diagonalise the matrix
            allowed_states=np.where(abs(E)<thresh)[0]
    project_vecs=V[:,allowed_states] # allowed states in computational basis
    return project_vecs # return projections

def precompute_projections(H_proj_s,num_step=1000): # function to precompute vectors to project into an allowed subspace
    '''
        H_proj_s is a function which returns a Hamiltonian the zero eigenspace of which is the projected subspace
    '''
    project_precompute=[None]*(num_step+1) # empty list of the correct size
    project_precompute[0]=project_H_allowed_state_space(H_proj_s(0)) # before stepping begins
    for i_step in range(num_step): # steps of the evolution
        s=i_step/(num_step-1) # s value
        project_precompute[i_step+1]=project_H_allowed_state_space(H_proj_s(s)) # calculate projection vectors
        #print(i_step)
    return project_precompute

def proj_space_sweep(H_proj_s,H_static,t_tot,start_state,measure_func,num_step=1000,project_precompute=None): # sweeps the projected space in the presence of a static Hamiltonian
    '''
        H_proj_s is a function which returns a Hamiltonian the zero eigenspace of which is the projected subspace
        H_static is a Hamiltonian within the full subspace which will be projected down to the time-dependent subspace
        t_tot is the total runtime
        start_state is the initial state for the evolution
        measure_func is a function which returns the desired properties for plotting
        num_step controls the number of steps in the simulation
        project_precompute is a list of pre-computed projection vectors to speed up computation, if None than projections are computed each time
        returns a list of the results of measure_func at each step
    '''
    if type(project_precompute)==type(None): # whether projections need to be computed
        project_vecs=project_H_allowed_state_space(H_proj_s(0)) # starting basis
    else:
        project_vecs=project_precompute[0] # first entry in pre-computed list
        num_steps=len(project_precompute)-1 # override number of steps if set
    state_vec=np.dot(project_vecs.conj().T,start_state) # initialise system
    data_list=[None]*num_step # list of outputs of measure func
    for i_step in range(num_step): # steps of the evolution
        s=i_step/(num_step-1) # s value
        project_vecs_old=copy.copy(project_vecs) # copy to build transition matrix
        if type(project_precompute)==type(None): # if not precomputed
            project_vecs=project_H_allowed_state_space(H_proj_s(s)) # new set of vectors for projection
        else: # load precomputed to save calculation time
            project_vecs=project_precompute[i_step+1]
        transition_mat=np.dot(project_vecs.conj().T,project_vecs_old) # matrix to transform basis
        state_vec=np.dot(transition_mat,state_vec)
        H_sub=np.dot(np.dot(project_vecs.conj().T,H_static),project_vecs) # project static Hamiltonian into subspace
        state_vec=sparse.linalg.expm_multiply(-1j*(t_tot/num_step)*H_sub,state_vec) # matrix exponetiation and multiplication
        data_list[i_step]=measure_func(state_vec,project_vecs,H_sub) # measure desired quantities
    return data_list
    
def project_sweep_fixed_matrix(M_s,M_static,start_state,measure_func,num_step=1000): # alternates between measuring if a variable is in the xi state and applying a fixed matrix
    '''
        M_s is a function of s which returns a dense matrix to be applied at each step of the evolution (typically a projection of xi, but doesn't have to be)
        M_static is an s-independent dense matrix to be applied at each step of the evolution, could be a projector or something else
        measure_func is a function to be applied at every step the output of which is returned
        num_step is the number of evenly spaced steps in theta which are taken
    '''
    state_vec=start_state # initial state of system
    data_list=[None]*num_step # list of outputs of measure func
    for i_step in range(num_step): # steps of the evolution
        s=i_step/(num_step-1) # s value
        state_vec=np.dot(M_static,np.dot(M_s(s),state_vec)) # perform matrix multiplication
        data_list[i_step]=measure_func(state_vec) # measure desired quantities
    return data_list


def var_neg_lists_to_clauses(var_lists,neg_lists,var_symbol="x",equation_label=None,clauses_per_line=3): # converts a list of variables and whether or not they are negated to a CNF expression
    '''
        var_lists is a list of lists, where each list within the list is a list of variable numbers
        neg_lists is a list of lists where each sub list is of the same legth as in var_lists and contains boolean expressions for whether or not they are negated
        equation_label is a label for the equation to be referred to in other parts of the document
        var_symbol is the symbol to use for each variable (defaults to x)
        clauses_per_line is the number of clauses to use before skipping to the next line
        returns a CNF_string which will be correct latex code when printed
    '''
    CNF_string="%%%%%%%% the following equation was automatically generated using a python script it may be easier to edit the source code than edit the latex directly \n"
    CNF_string=CNF_string+"\\begin{align} \n \t" # start environment
    for i_clause in range(len(var_lists)): # loop through clauses
        CNF_string=CNF_string+"\\left( " # opening bracket for specific clause
        for i_var in range(len(var_lists[i_clause])):
            if neg_lists[i_clause][i_var]: # if negated
                CNF_string=CNF_string+"\\neg " # negation symbol
            CNF_string=CNF_string+var_symbol+"_{"+str(var_lists[i_clause][i_var])+"} " # write symbol with subscript
            if not i_var==(len(var_lists[i_clause])-1): # check if this is the end of the clause or needs to be proceeded by a vee
                CNF_string=CNF_string+" \\vee " # add vee to string
        CNF_string=CNF_string+" \\right)" # close parenthensis
        if not i_clause==(len(var_lists)-1): # if not the final clause add a wedge
            CNF_string=CNF_string+" \\wedge " # add wedge to string
            if (i_clause+1)%clauses_per_line==0: # if we need a line break
                CNF_string=CNF_string+" \\nonumber \\\\ \n \t" # break line and tab
    if not type(equation_label)==type(None): # if label is to be added
        CNF_string=CNF_string+"\n \\label{"+equation_label+"}" # add equation label
    CNF_string=CNF_string+"\n \end{align} \n %%%%%%% end automatically generated latex"
    return CNF_string # return latex string

def list_of_lists_to_latex(list_of_lists,equation_label=None,lists_per_line=9): # converts a list of lists to a latex align environment
    '''
        list_of_lists is a list of lists which does not have to contain strings
        equation_label is a label for the equation to be referred to in other parts of the document
        lists_per_line is the number of lists which display on each line
        returns a string which prints in an align environment in latex
    '''
    lol_string="%%%%%%%% the following equation was automatically generated using a python script it may be easier to edit the source code than edit the latex directly \n"
    lol_string=lol_string+"\\begin{align} \n \t" # start environment
    lol_string=lol_string+"[ " # opening charater for list
    for i_list1 in range(len(list_of_lists)): # loop through list of lists
        lol_string=lol_string+"[ " # opening charater for list
        for i_list2 in range(len(list_of_lists[i_list1])): # loop through lists
            lol_string=lol_string+str(list_of_lists[i_list1][i_list2]) # convert element to string and add
            if not i_list2==(len(list_of_lists[i_list1])-1): # check if this is the end of the list, if not a comma is needed
                lol_string=lol_string+" , " # add comma to string
        lol_string=lol_string+"] " # closing charater for list
        if not i_list1==(len(list_of_lists)-1): # check if this is the end of the list of lists, add a comma if not
            lol_string=lol_string+" , " # add comma to string
            if (i_list1+1)%lists_per_line==0: # if we need a line break
                lol_string=lol_string+" \\nonumber \\\\ \n \t" # break line and tab
    lol_string=lol_string+" ]" # closing charater for list
    if not type(equation_label)==type(None): # if label is to be added
        lol_string=lol_string+"\n \\label{"+equation_label+"}" # add equation label
    lol_string=lol_string+"\n \end{align} \n %%%%%%% end automatically generated latex"
    return lol_string

            
