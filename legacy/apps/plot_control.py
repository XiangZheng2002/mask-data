import numpy as np
import matplotlib.pyplot as plt
import os
import pdb


def plot_control(control_npy_full, save_path):
    size_ = 64
    x, y = np.meshgrid(np.linspace(0, size_-1, size_), np.linspace(0, size_-1, size_))
    for frame in range(65):
        control_npy = control_npy_full[:,:,:,frame]
        pic_path = os.path.join(save_path, f'{frame}.png')
        xvel1 = np.zeros([64]*2)
        yvel1 = np.zeros([64]*2)

        xvel1 = control_npy[:, :, 0]
        yvel1 = control_npy[:, :, 1]

        fig, ax1 = plt.subplots(figsize=(8, 8)) 
        quiver = ax1.quiver(x, y, xvel1, yvel1, scale=5, scale_units='inches')
        ax1.set_title(f'Control: {save_path} Frame: {frame}', fontsize=12)
    
        plt.savefig(pic_path, dpi=50)
        plt.close(fig)




if __name__ == '__main__':
    data_dir_path = './mask-data-1117/gt_50_100_20_1_3'
    save_dir = './control_1117'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    for file in os.listdir(data_dir_path):
        sim_path = os.path.join(data_dir_path, file)
        control_path = os.path.join(sim_path, 'Control.npy')
        data = np.load(control_path)
        save_path = os.path.join(save_dir, file)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plot_control(data, save_path)