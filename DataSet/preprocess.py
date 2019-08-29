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
            converted_data_seg = np.zeros((data_seg.shape[0], 6))

            data_seg_x = (data_seg[:, 1]+data_seg[:, 3]) / 2
            data_seg_y = (data_seg[:, 2]+data_seg[:, 4]) / 2
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

        if not os.path.exists(train_path): os.makedirs(train_path)
        if not os.path.exists(test_path): os.makedirs(test_path)

        train_count = len([name for name in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, name))])
        test_count = len([name for name in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, name))])

        train_name = os.path.join(train_path, 'data'+str(train_count))
        test_name = os.path.join(test_path, 'data'+str(test_count))

        np.savetxt(train_name, train_data, delimiter=',')
        np.savetxt(test_name, test_data, delimiter=',')


    def pipeline(self):
        self.data_files()
        for file in self.files:
            converted_data = self.data_convert(file)
            self.data_save(converted_data)
    
'''
NGSIM US-101 Dataset
'''
class NGSIMDataset():
    def __init__(self, dset_path, save_path='dataset'):
        self.dset_path = dset_path
        self.save_path = save_path
        self.base_path = '.'
        self.files = []

    
    def data_files(self):
        path = os.path.join(self.dset_path, self.base_path)
        for name in os.listdir(path):
            if os.path.isfile(name):
                file = os.path.join(path, name)
                self.files.append(file)
    

    def data_convert(self, file):
        raw_data = np.loadtxt(file, delimiter=',', usecols=(0, 1, 2, 3), skiprows=1)
        converted_data = []

        data = raw_data[np.argsort(raw_data[:, 0])]
        agent_ids = np.unique(data[:, 0])

        for agent_id in agent_ids:
            data_seg = data[data[:, 0] == agent_id, :]
            converted_data_seg = np.zeros((data_seg.shape[0], 6))

            data_seg_x = data_seg[:, 3]
            data_seg_y = data_seg[:, 2]
            data_seg_vx = data_seg_x[1:]-data_seg_x[:-1]
            data_seg_vy = data_seg_y[1:]-data_seg_y[:-1]
            data_seg_vx, data_seg_vy = np.append(data_seg_vx, 0.0), np.append(data_seg_vy, 0.0)

            converted_data_seg[:, 0] = data_seg[:, 1]
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

        train_path = os.path.join(self.save_path, 'NGSIMDataset', 'train')
        test_path = os.path.join(self.save_path, 'NGSIMDataset', 'test')

        if not os.path.exists(train_path): os.makedirs(train_path)
        if not os.path.exists(test_path): os.makedirs(test_path)

        train_count = len([name for name in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, name))])
        test_count = len([name for name in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, name))])

        train_name = os.path.join(train_path, 'data'+str(train_count))
        test_name = os.path.join(test_path, 'data'+str(test_count))

        np.savetxt(train_name, train_data, delimiter=',')
        np.savetxt(test_name, test_data, delimiter=',')

    
    def pipeline(self):
        self.data_files()
        for file in self.files:
            converted_data = self.data_convert(file)
            self.data_save(converted_data)


'''
GTA Dataset
'''
class GTADataset():
    def __init__(self, dset_path, save_path='dataset', tag='GTAS', full_tag='straight', number=6):
        self.dset_path = dset_path
        self.save_path = save_path
        self.tag = tag
        self.full_tag = full_tag
        self.number = number
        self.files = []

    
    def data_files(self):
        for idx in range(self.number):
            dir_path = os.path.join(self.dset_path, self.tag, '['+str(idx)+']'+self.full_tag)
            for name in os.listdir(dir_path):
                file = os.path.join(dir_path, name)
                if os.path.isfile(file):
                    self.files.append(file)
    

    def data_convert(self, file):
        raw_data = np.loadtxt(file, delimiter=',', usecols=(0, 1, 2, 3))
        converted_data = []

        data = raw_data[np.argsort(raw_data[:, 1])]
        agent_ids = np.unique(data[:, 1])

        for agent_id in agent_ids:
            data_seg = data[data[:, 1] == agent_id, :]
            converted_data_seg = np.zeros((data_seg.shape[0], 6))

            data_seg_x = data_seg[:, 2]
            data_seg_y = data_seg[:, 3]
            data_seg_vx = data_seg_x[1:]-data_seg_x[:-1]
            data_seg_vy = data_seg_y[1:]-data_seg_y[:-1]
            data_seg_vx, data_seg_vy = np.append(data_seg_vx, 0.0), np.append(data_seg_vy, 0.0)

            converted_data_seg[:, 0] = data_seg[:, 0]
            converted_data_seg[:, 1] = data_seg[:, 1]
            converted_data_seg[:, 2] = data_seg_x
            converted_data_seg[:, 3] = data_seg_y
            converted_data_seg[:, 4] = data_seg_vx
            converted_data_seg[:, 5] = data_seg_vy

            converted_data.append(converted_data_seg)
        converted_data = np.concatenate(converted_data, axis=0)

        converted_data = converted_data[np.argsort(converted_data[:, 0])]

        return converted_data

    
    def data_save(self, converted_data):
        bound = int(converted_data.shape[0]*0.6)
        train_data = converted_data[:bound, :]
        test_data = converted_data[bound:, :]

        train_path = os.path.join(self.save_path, 'GTADataset', self.tag, 'train')
        test_path = os.path.join(self.save_path, 'GTADataset', self.tag, 'test')

        if not os.path.exists(train_path): os.makedirs(train_path)
        if not os.path.exists(test_path): os.makedirs(test_path)

        train_count = len([name for name in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, name))])
        test_count = len([name for name in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, name))])

        train_name = os.path.join(train_path, 'data'+str(train_count))
        test_name = os.path.join(test_path, 'data'+str(test_count))

        np.savetxt(train_name, train_data, delimiter=',')
        np.savetxt(test_name, test_data, delimiter=',')


    def pipeline(self):
        self.data_files()
        for file in self.files:
            converted_data = self.data_convert(file)
            self.data_save(converted_data)

# start preprocess
pre_dset = GTADataset('/mnt/Dataset/TrajDset/GTA', save_path='dataset', tag='GTAS', full_tag='straight', number=6)
pre_dset.pipeline()
