import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
import zeno_constraint as zc
import matplotlib.pyplot as plt
import time

Z3=np.zeros([3,3])
Z3[1,1]=1
Z3[2,2]=-1
Z2=np.zeros([2,2])
Z2[0,0]=1
Z2[1,1]=-1
X2=np.zeros([2,2])
X2[0,1]=1
X2[1,0]=1
h_3h5=1*np.ones(5)
J_3h5=np.zeros([5,5])
for i_J in range(5):
    for j_J in range(i_J+1,5):
        J_3h5[i_J,j_J]=1
h_5=np.load('random_h5.npy')


str_list=np.logspace(-1,3,200)
t_anneal_list=[5,10,100]
final_success_probs_nI2=np.zeros([len(str_list),len(t_anneal_list)])
for i_const_str in range(len(str_list)):
    t_start=time.time()
    constraint_str=str_list[i_const_str]
    #min_cstr=min(np.diag(zc.construct_quadratic_Hamiltonian(Z3,J_3h5,h_3h5).todense()))
    #min_h=min(np.diag(zc.construct_quadratic_Hamiltonian(Z3,np.zeros([5,5]),h_5).todense()))
    H_Ising=zc.construct_quadratic_Hamiltonian(Z2,np.zeros([5,5]),h_5)
    #H_s=lambda s: 1j*abs(constraint_str*min_cstr+min_h)*(zc.multi_qubit_rotate_disp_ham(s*np.pi/2,5))-0.1*zc.multi_var_u_offset_ham(5)+H_Ising
    #H_s=lambda s: 1j*10*abs(constraint_str*min_cstr)*(zc.multi_qubit_rotate_disp_ham(s*np.pi/2,5))-abs(constraint_str*min_cstr+1)*zc.multi_var_u_offset_ham(5)+H_Ising
    H_proj_s_3h5=lambda s: zc.project_forbidden_weights(5,lambda x:x==3,qubit=True)
    H_s=lambda s: constraint_str*H_proj_s_3h5(s)-(1-s)*zc.construct_quadratic_Hamiltonian(X2,np.zeros([5,5]),np.ones(5))+s*H_Ising
    E_V_precomp_list=zc.precompute_E_V_eigh(H_s,1000)
    print('precomp time (2d)= '+str(time.time()-t_start))
    print(i_const_str)
    print('constraint strength= '+str(constraint_str))
    t_start=time.time()
    for i_anneal_time in range(len(t_anneal_list)):
        sat_state=np.zeros(2**5)
        sat_state[7]=1
        t_tot=t_anneal_list[i_anneal_time]
        start_state=np.ones(2**5)
        start_state=start_state/linalg.norm(start_state)
        measure_func=lambda state_vec: [linalg.norm(state_vec)**2,abs(np.dot(sat_state,state_vec))**2,abs(np.dot(start_state,state_vec))**2]
        method='eigh'
        data_list=zc.zeno_sweep_H_expm_mult(H_s,t_tot,start_state,measure_func,method=method,E_V_precomp_list=E_V_precomp_list)
        final_success_probs_nI2[i_const_str,i_anneal_time]=data_list[-1][1]
        print(i_anneal_time)
        sat_prob_list=[[]+[data[1]] for data in data_list]
        #plt.plot(sat_prob_list)
        #plt.show()
        print('total sweep calculation time= '+str(time.time()-t_start))
    np.save('adiabat_2_sweep_constrain.npy',{'t_anneal_list':t_anneal_list,'final_success_probs': final_success_probs_nI2,'constraint_str_list':str_list})
    #plt.semilogx(str_list,final_success_probs_nI2)
    #plt.show()

#str_list=np.logspace(-1,3,200)
#t_anneal_list=[5,10,100]
final_success_probs_nI=np.zeros([len(str_list),len(t_anneal_list)])
for i_const_str in range(len(str_list)):
    t_start=time.time()
    constraint_str=str_list[i_const_str]
    H_Ising=zc.construct_quadratic_Hamiltonian(Z3,np.zeros([5,5]),h_5)
    #H_s=lambda s: 1j*abs(constraint_str*min_cstr+min_h)*(zc.multi_qubit_rotate_disp_ham(s*np.pi/2,5))-0.1*zc.multi_var_u_offset_ham(5)+H_Ising
    #H_s=lambda s: 1j*10*abs(constraint_str*min_cstr)*(zc.multi_qubit_rotate_disp_ham(s*np.pi/2,5))-abs(constraint_str*min_cstr+1)*zc.multi_var_u_offset_ham(5)+H_Ising
    H_proj_s_3h5=lambda s: zc.project_forbidden_weights(5,lambda x:x==3)+1j*zc.multi_qubit_rotate_disp_ham(s*np.pi/2,5)
    H_s=lambda s: constraint_str*H_proj_s_3h5(s)-zc.multi_var_u_offset_ham(5)+H_Ising
    E_V_precomp_list=zc.precompute_E_V_eigh(H_s,1000)
    print('precomp time= '+str(time.time()-t_start))
    print(i_const_str)
    print('constraint strength= '+str(constraint_str))
    t_start=time.time()
    for i_anneal_time in range(len(t_anneal_list)):
        sat_state=np.zeros(3**5)
        sat_state[134]=1
        t_tot=t_anneal_list[i_anneal_time]
        start_state=np.zeros(3**5)
        start_state[0]=1
        measure_func=lambda state_vec: [linalg.norm(state_vec)**2,abs(np.dot(sat_state,state_vec))**2,abs(np.dot(start_state,state_vec))**2]
        method='eigh'
        data_list=zc.zeno_sweep_H_expm_mult(H_s,t_tot,start_state,measure_func,method=method,E_V_precomp_list=E_V_precomp_list)
        final_success_probs_nI[i_const_str,i_anneal_time]=data_list[-1][1]
        print(i_anneal_time)
        sat_prob_list=[[]+[data[1]] for data in data_list]
        print('total sweep calculation time= '+str(time.time()-t_start))
    np.save('adiabat_3_sweep_constrain.npy',{'t_anneal_list':t_anneal_list,'final_success_probs': final_success_probs_nI,'constraint_str_list':str_list})

