import argparse
import datetime
import matplotlib.pylab as plt
import numpy as np
import pdb
import time
import pdb
from datetime import datetime
import os
from evaluate_solver_original_64 import *
import multiprocessing

current_time = datetime.now()
current_month = current_time.month
current_day = current_time.day 
current_hour = current_time.hour 
current_minute = current_time.minute 
global time_str
time_str = f"{current_month}{current_day}_{current_hour}{current_minute}"



def test_files_ground_solver_save2(space_length, time_sample_rate, file_store_path_, time_str, branch_num, per_branch, per_timelength):
    # file_store_path = os.path.join(file_store_path_, time_str)
    smoke_out_list = []
    outlier_dict = {}
    # for sim_id in os.listdir(file_store_path_):
    for id in range(branch_num*per_branch,(branch_num+1)*per_branch):
        if id < 10:
            sim_id = f'sim_00000{id}'
        elif id < 100:
            sim_id = f'sim_0000{id}'
        elif id < 1000:
            sim_id = f'sim_000{id}'
        elif id < 10000:
            sim_id = f'sim_00{id}'
        else:
            sim_id = f'sim_0{id}'
        sim_path = os.path.join(file_store_path_, sim_id)
        des_path = os.path.join(sim_path, 'Density.npy')
        vel_path = os.path.join(sim_path, 'Velocity.npy')
        control_path = os.path.join(sim_path, 'Control.npy')
        smoke_path = os.path.join(sim_path, 'Smoke.npy')
        des = np.load(des_path) # [128,128,1,257] # (64, 64, 1, 33)
        vel = np.load(vel_path) # [128,128,2,257]
        control = np.load(control_path) # [128,128,2,257]
        smoke = np.load(smoke_path)[:-1,:] # (257, 8)

        # Let the first step be the inital step
        init_velocity = vel[:,:,:,0]
        init_density = des[:,:,0,0]

        c1 = control[:,:,0,:-1]
        c1 = np.transpose(c1, (2,0,1))
        c2 = control[:,:,1,:-1]
        c2 = np.transpose(c2, (2,0,1))

        sim1 = init_sim()
        density_solver_from1, zero_densitys_solver_from1, velocitys_solver_from1, _, _, smoke_target_solver, smoke_out_solver, density_sum_record, remain_density_qualify= solver_debug(sim1, vel[:,:,:,0], des[:,:,0,0], c1, c2, per_timelength)
        
        # (256, 128, 128), (256, 128, 128), (256, 128, 128, 2)

        csv_header = 'sim_id,ground target,ground sum,ground rate,solver target,solver sum,solver rate,target error,sum error,absolute smoke error,relative smoke error,remain smoke qualify'
        smoke_out_2 = [0] * 12
        # sim_id
        smoke_out_2[0] = id

        # ground truth
        smoke_out_2[1] = smoke[-1][1] # target smoke
        smoke_out_2[2] = np.sum(smoke[-1]) # density sum
        smoke_out_2[3] = smoke[-1][1]/ np.sum(smoke[-1]) # smoke rate

        # solver
        smoke_out_2[4] = smoke_target_solver[-1*time_sample_rate] # target smoke
        smoke_out_2[5] = smoke_target_solver[-1*time_sample_rate] / smoke_out_solver[-1*time_sample_rate] # density smoke
        smoke_out_2[6] = smoke_out_solver[-1*time_sample_rate]  # smoke rate

        # error
        smoke_out_2[7] = smoke[-1][1] - smoke_out_2[4] # target error
        smoke_out_2[8] = smoke_out_2[2] - smoke_out_2[5] # sum error
        smoke_out_2[9] = abs(smoke_out_2[3]-smoke_out_2[6]) # absolute smoke error
        smoke_out_2[10] = abs(smoke_out_2[3]-smoke_out_2[6]) / smoke_out_2[3] # relative smoke error

        smoke_out_2[11] = remain_density_qualify
        smoke_out_list.append(smoke_out_2)

        # if smoke_out_2[10] > 0.1:
        if True:
            ground_truth = [np.transpose(des[:,:,0,:-1], (2,0,1)), np.transpose(vel[:,:,:,:-1], (3,0,1,2))]

            # if space_length==64:
            #     solver_solution = [density_solver_from1[::time_sample_rate,:,:],velocitys_solver_from1[::time_sample_rate,:,:,:]]
            # elif space_length==128:
            #     solver_solution = [density_solver_from1[::time_sample_rate,::2,::2],velocitys_solver_from1[::time_sample_rate,::2,::2,:]]
            solver_solution = [density_solver_from1[::time_sample_rate,:,:], velocitys_solver_from1[::time_sample_rate,:,:]]

            smoke_target_ground = np.around(smoke[:,1], decimals=4)
            density_sum_ground = np.around(np.sum(smoke, axis=1)[:], decimals=4)
            smoke_rate_ground = smoke_target_ground / density_sum_ground

            ground_packet = [smoke_target_ground, density_sum_ground, smoke_rate_ground]

            smoke_target_solver = np.around(smoke_target_solver[::time_sample_rate], decimals=4)
            density_sum_solver = np.around(density_sum_record[::time_sample_rate], decimals=4)
            smoke_rate_solver = np.around(smoke_out_solver[::time_sample_rate], decimals=4)

            solver_packet = [smoke_target_solver, density_sum_solver, smoke_rate_solver]

            outlier_dict[sim_id] = [ground_packet, solver_packet, ground_truth, solver_solution]

        print(f"Control of {sim_id} Down!")

    smoke_out_list = np.stack(smoke_out_list)
    
    # debug_smoke_path = f'./debug-64-data/smoke_out_compare/{time_str}.txt'
    file_path = f'./{save_dir}/smoke_out_compare/percent_{time_str}.csv'
    mode = 'a' if os.path.exists(file_path) else 'w'

    with open(file_path, mode) as f:
        if mode == 'w':
            f.write(csv_header + '\n')
        np.savetxt(f, smoke_out_list, delimiter=',', fmt=['%d'] + ['%.6f']*11)

    if not outlier_dict:
        return True
    else:
        return outlier_dict



def get_initial_diff(file_store_path_, time_str, branch_num, per_branch):
    # file_store_path = os.path.join(file_store_path_, time_str)
    # for sim_id in os.listdir(file_store_path_):
    for id in range(branch_num*per_branch,(branch_num+1)*per_branch):
        initial_sum_np = np.zeros((1,2), dtype=float)
        if id < 10:
            sim_id = f'sim_00000{id}'
        elif id < 100:
            sim_id = f'sim_0000{id}'
        sim_path = os.path.join(file_store_path_, sim_id)
        des_path = os.path.join(sim_path, 'Density.npy')
        vel_path = os.path.join(sim_path, 'Velocity.npy')
        control_path = os.path.join(sim_path, 'Control.npy')
        smoke_path = os.path.join(sim_path, 'Smoke.npy')
        des = np.load(des_path) # [128,128,1,257] # (64, 64, 1, 33)
        vel = np.load(vel_path) # [128,128,2,257]
        control = np.load(control_path) # [128,128,2,257]
        smoke = np.load(smoke_path) # (257, 8)

        # Let the first step be the inital step
        init_velocity = vel[:,:,:,0]
        init_density = des[:,:,0,0]

        csv_header = 'sim_id,initial density'
        initial_sum_np[0][0] = id
        initial_sum_np[0][1] = float(np.sum(init_density))

        file_path = './temp/inital_sum.csv'

        mode = 'a' if os.path.exists(file_path) else 'w'

        with open(file_path, mode) as f:
            if mode == 'w':
                f.write(csv_header + '\n')
            # np.savetxt(f, initial_sum_np, delimiter=',', fmt=['%d'] + ['%.6f'])
            np.savetxt(f, initial_sum_np, delimiter=',', fmt='%.6f')


def cal_smoke_and_plot(time_sample_rate, space_length, original_timelength, data_path, time_str, pic_savepath, branch_num, per_branch):
    density_path = os.path.join(pic_savepath, 'density/')
    velocity_path = os.path.join(pic_savepath, 'velocity/')

    per_timelength = original_timelength

    flag = test_files_ground_solver_save2(space_length, time_sample_rate, data_path, time_str, branch_num, per_branch, per_timelength)
    
    if not flag==True:
        outlier_dict = flag
        for key, value in outlier_dict.items():
            gif_density_64_debug(outlier_value=value,pic_dir=density_path,sim_id=key)
            plot_vector_field_64_debug(ground_packet=value[0],solver_packet=value[1],vel_ground=value[2][1],vel_solver=value[3][1],vel_pic_path_=velocity_path,sim_id=key)
            print(f'{key} DOWN!')


def turn_npy_to_csv(path):
    data = np.load(path)
    np.savetxt(f'./debug-64-data/smoke_out_compare/{time_str}.csv', data, delimiter=',')
    print('down!')




# if __name__ == "__main__":
#     test_from_read()
#     test_files_ground_solver_dens('./debug-data/0628_0023')
#     time_str = '0628_1550'
#     time_str = '0629_1754'
#     test_files_ground_solver_dens(f'./debug-data/groundtruth_0629_1754', time_str)
#     test_files_ground_solver_save('./debug-data/', time_str)
#     test_files_ground_solver_save2(f'./debug-data/groundtruth_0629_1754', '0629_1754')
#     turn_npy_to_csv(f'./debug-64-data/smoke_out_compare/0629_1754.txt.npy')
#     cal_smoke_and_plot(data_path='./debug-data/groundtruth_0701_smoke_128', time_str='0701_smoke_128',pic_savepath='./debug-gif/0701')

# if __name__ == "__main__":
#     branch_list = np.arange(16)
#     per_branch = 5
#     per_scenelength = 128
#     space_length = 64
#     time_sample_rate = int(per_scenelength / 32)
#     # data_path = './debug-data/groundtruth_0701_initial_pos'
#     # data_path = './debug-data/groundtruth_0701_82'
#     # data_path = './debug-data/groundtruth_0701_2245'
#     data_path = './debug-data/groundtruth_0702_perscenelength_128'
#     # data_path = './debug-data/groundtruth_0702_perscenelength_64'
#     # data_path = './debug-data/groundtruth_0702_32(32)_128_128'




#     time_str = '0702_perscenelength_128'
#     pic_savepath=f'./Debug_0701/debug-gif/{time_str}'
#     density_path = os.path.join(pic_savepath, 'density/')
#     velocity_path = os.path.join(pic_savepath, 'velocity/')

#     if not os.path.exists(density_path):
#         os.makedirs(density_path)
#     if not os.path.exists(velocity_path):
#         os.makedirs(velocity_path)

#     if not os.path.exists(f'./Debug_0701/smoke_out_compare/'):
#         os.makedirs(f'./Debug_0701/smoke_out_compare/')

#     with multiprocessing.Pool(len(branch_list)) as pool:
#         args_func = [(space_length, time_sample_rate, data_path,time_str,pic_savepath,branch_num, per_branch) for branch_num in branch_list]
#         pool.starmap(cal_smoke_and_plot, args_func)

    # cal_smoke_and_plot(time_sample_rate,data_path,time_str,pic_savepath,2,1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_savepath", type=str, help='dataset location')
    parser.add_argument("--branch_begin", type=str, help='branch begin number')
    parser.add_argument("--branch_end", type=str, help='branch end number')
    parser.add_argument("--original_timelength",type=str,help='timelength before downsample')
    parser.add_argument("--time_length", type=str, help='downsample time to length n')
    parser.add_argument("--space_length", type=str, help='downsample space to length n')
    parser.add_argument("--per_branch", type=str, help='mission for per branch')
    parser.add_argument("--save_dir", type=str, help='place to save the result')


    args = parser.parse_args()
    data_path = args.data_savepath

    begin_no = int(args.branch_begin)
    end_no = int(args.branch_end)
    branch_list = np.arange(begin_no,end_no)
    time_length = int(args.time_length)
    space_length = int(args.space_length)
    per_branch = int(args.per_branch)
    original_timelength = int(args.original_timelength)
    global save_dir
    save_dir = args.save_dir

    branch_list = np.arange(begin_no, end_no)
    # time_str = data_path.split('/')[-1][11:]
    time_str = data_path.split('/')[-1]
    pic_savepath=f'./{save_dir}/debug-gif/{time_str}'

    density_path = os.path.join(pic_savepath, 'density/')
    velocity_path = os.path.join(pic_savepath, 'velocity/')
    time_sample_rate = int(original_timelength / time_length)


    if not os.path.exists(density_path):
        os.makedirs(density_path)
    if not os.path.exists(velocity_path):
        os.makedirs(velocity_path)

    if not os.path.exists(f'./{save_dir}/smoke_out_compare/'):
        os.makedirs(f'./{save_dir}/smoke_out_compare/')

    with multiprocessing.Pool(len(branch_list)) as pool:
        args_func = [(time_sample_rate, space_length, original_timelength, data_path,time_str,pic_savepath,branch_num, per_branch) for branch_num in branch_list]
        pool.starmap(cal_smoke_and_plot, args_func)

    # cal_smoke_and_plot(time_sample_rate, space_length, original_timelength, data_path,time_str,pic_savepath,1,1)
