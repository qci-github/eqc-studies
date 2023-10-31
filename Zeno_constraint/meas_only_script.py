import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
import zeno_constraint as zc
import matplotlib.pyplot as plt
import time



t_start=time.time()
(var_lists,neg_lists)=np.load('unique_3_sat_5_1690552185.884515.npy')[()]
H_sat_diag=zc.build_sat_Ham_three_diag(var_lists,neg_lists,5)
num_step_list=[10,10**2,10**3,10**4,10**5,10**6,10**7]
data_list_list=[None]*len(num_step_list)
steps_list_list=[None]*len(num_step_list)
sat_state_P=np.diag(np.array(H_sat_diag==0,dtype=float))
P_s=lambda s: zc.multi_qubit_measure_project_xi(s*np.pi/2,5)
sat_state=np.zeros(3**5)
sat_state[121]=1
start_state=np.zeros(3**5)
start_state[0]=1
measure_func=lambda state_vec: [linalg.norm(state_vec)**2,abs(np.dot(sat_state,state_vec))**2,abs(np.dot(start_state,state_vec))**2]
for i_num_step in range(len(num_step_list)):
    num_step=num_step_list[i_num_step]
    data_list=zc.project_sweep_fixed_matrix(P_s,sat_state_P,start_state,measure_func,num_step=num_step)
    if len(data_list)>1000:
        steps_list_list[i_num_step]=list(range(0,len(data_list),int(len(data_list)/1000)))
        data_list_list[i_num_step]=[data_list[i_skip] for i_skip in range(0,len(data_list),int(len(data_list)/1000))]
    else:
        steps_list_list[i_num_step]=list(range(len(data_list)))
        data_list_list[i_num_step]=data_list
    print(num_step)
    print(time.time()-t_start)
np.save('sat_measure_only.npy',{'num_step_list': num_step_list,'steps_list_list':steps_list_list,'data_list_list':data_list_list})
