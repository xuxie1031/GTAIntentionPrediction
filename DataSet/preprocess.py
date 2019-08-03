'''
format orginal dataset data into (frame, id, local x, local y, local vx, local vy .etc)
divide dataset data into 'train' / 'test'
'''

import numpy as np
import os

'''
Stanford Drone Dataset
'''

class DroneDataset():
    def __init__(self, dset_path, save_path='dataset'):
        self.dset_path = dset_path
        self.save_path = save_path
        self.base_path = 'annotations'
        self.categories = {'bookstore':7, 'coupa':4, 'deathCircle':5, 'gates':9, 'hyang':15, 'little':4, 'nexus':12, 'quad':4}
        self.files = []


    def data_files(self):
        for name, num in self.categories.items():
            for i in range(num):
                file = os.path.join(self.dset_path, self.base_path, name, 'video'+str(i), 'annotations.txt')
                self.files.append(file)


    def data_convert(self, file):
        raw_data = np.loadtxt(file, delimiter=' ', usecols=(0, 1, 2, 3, 4, 5))
        converted_data = []

        data = raw_data[np.argsort(raw_data[:, 0])]
        agent_ids = np.unique(data[:, 0])

        for agent_id in agent_ids:
            data_seg = data[data[:, 0] == agent_id, :]
            converted_data_seg = np.zeros(data_seg.shape[0], 6)

            data_seg_x = (data_seg[:, 1]+data_seg[:, 2]) / 2
            data_seg_y = (data_seg[:, 3]+data_seg[:, 4]) / 2
            data_seg_vx = data_seg_x[1:]-data_seg_x[:-1]
            data_seg_vy = data_seg_y[1:]-data_seg_y[:-1]
            data_seg_vx, data_seg_vy = np.append(data_seg_vx, 0.0), np.append(data_seg_vy, 0.0)
            
            converted_data_seg[:, 0] = data_seg[:, 5]
            converted_data_seg[:, 1] = data_seg[:, 0]
            converted_data_seg[:, 2] = data_seg_x
            converted_data_seg[:, 3] = data_seg_y
            converted_data_seg[:, 4] = data_seg_vx
            converted_data_seg[:, 5] = data_seg_vy

            converted_data.append(converted_data_seg)
        converted_data = np.concatenate(converted_data, axis=0)

        converted_data = converted_data[np.argsort(converted_data[:, 0])]

        return converted_data


    def data_save(self, converted_data):
        bound = int(converted_data.shape[0]*0.7)
        train_data = converted_data[:bound, :]
        test_data = converted_data[bound:, :]

        train_path = os.path.join(self.save_path, 'DroneDataset', 'train')
        test_path = os.path.join(self.save_path, 'DroneDataset', 'test')

        train_count = len([name for name in os.listdir(train_path) if os.path.isfile(name)])
        test_count = len([name for name in os.listdir(test_path) if os.path.isfile(name)])

        train_name = os.path.join(train_path, 'data'+str(train_count))
        test_name = os.path.join(test_path, 'data'+str(test_count))

        np.savetxt(train_name, train_data, delimiter=',')
        np.savetxt(test_name, test_data, delimiter=',')


    def pipeline(self):
        for file in self.files:
            converted_data = self.data_convert(file)
            self.data_save(converted_data)
    

class NGSIMDataset():
    def __init__(self):
        pass