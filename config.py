from pprint import pprint
import torch
import numpy as np
import os
import time

class Config:
    #########################################
    # 1. paths that include the training image and label
    all_paths='/media/hp/study/brain_train_data'

    #########################################
    # decide whether need to generate samples or not
    random_sample_flag = 0
    dataset_prefix = 'DHPR_Selected'

    # 2. training paramter setting
    patch_size = [120, 120, 120]
    num_workers = 5
    use_cuda=True
    thred = 32

    train_augument=True
    limited_range=[0,1500]
    train_shuffile=True
    train_batch=3
    train_epoch=100
    train_plotfreq=6

    # 3. model setting
    model_choice='VoxResNet_3D'
    in_dim=1
    out_dim=2
    save_parameters_name = dataset_prefix
    load_state=True

    # 4. optimizer parameter setting
    optimizer='SGD'
    lr=0.01
    momentum=0.9
    weight_decay=0.0003
    scheduler='StepLR'
    step_size=1
    gamma=0.5

    # 5. record the result
    log_path='result'
    log_name = os.path.join(log_path, 'log_{}.txt'.format(save_parameters_name))

    # 6. val parameters
    val_run=True
    val_plotfreq=2
    save_img=True
    val_batch=3

    # 7. test setting
    test_paths='/media/hp/work/Neuron_Data/Cmp_1200/cmp_diff method/007'
    test_batch = 3
    test_patch_size=np.array([120,120,120])
    test_overlap=np.array([10,10,10])



    ###############: print the informations
    def _parse(self):
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}



###### build the instance
opt = Config()

