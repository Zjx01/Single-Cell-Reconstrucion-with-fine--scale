"""This function is used to generate random samples with same size for training and validation
"""

from dataset.tools import *
from utils.tools import remake_dirs
import tifffile
import random
from config import opt


def GenerateDataList_Ind(Image_Path,selected_ind1,save_prex):
    """gennerate the namelist for train and val"""
    image_path=os.path.join(Image_Path,'image')
    label_path=os.path.join(Image_Path,'label')
    selected_ind1=selected_ind1.flatten()

    # image and label had corresponding names
    image_list=[image_path+'/{}.tif'.format(name_num) for name_num in selected_ind1]
    label_list=[label_path+'/{}.tif'.format(name_num) for name_num in selected_ind1]

    # save the list name
    WriteList2Txt(save_prex+'_image_list.txt',image_list)
    WriteList2Txt(save_prex+'_label_list.txt',label_list)


def RandomSelectPatches(patch_size,patch_num,prefix,threshold1=1200,phase='train'):
    # read the image and label list
    image_list_name = prefix+'/' + phase + '_image_list.txt'
    label_list_name = prefix+'/' + phase + '_label_list.txt'

    image_list = ReadTxt2List(image_list_name)
    label_list = ReadTxt2List(label_list_name)

    new_patch_index=[]

    # decide to generate new patch
    count_num=0

    while count_num<patch_num:
        # random get image index
        inum = random.randint(0,len(image_list)-1)

        # calculate whether fit the choice
        label1 = tifffile.imread(label_list[inum])
        label_patch1,position1=RandomPatches(label1,patch_size)
        label_patch_indnum = LabelIndexNum(label_patch1)

        # fit a predefined rule
        if label_patch_indnum>=threshold1:
            new_position=np.hstack((position1,inum))
            new_patch_index.append(new_position)
            count_num+=1
            print('-----{} generated_num:{}'.format(phase,count_num))

            # save the index list
            np.save(prefix+'/{}_random_patch_ind.npy'.format(phase), new_patch_index)


def Generate_Random_Samples(opt):
    # 1. read the index of selected images by DHPR selection
    selected_index=np.load(opt.all_paths+'/selectd_rule1.npy')

    # 2. generate the namelist of images and labels for training
    data_path='dataset/'+opt.dataset_prefix
    remake_dirs(data_path)

    GenerateDataList_Ind(opt.all_paths, selected_index, data_path+'/train')
    GenerateDataList_Ind(opt.all_paths, selected_index[:50], data_path + '/val')

    # 3. begin to generate random patches from these selected images for training
    RandomSelectPatches(opt.patch_size, 300*50, data_path, threshold1=1000, phase='train')
    RandomSelectPatches(opt.patch_size, 50*50, data_path, threshold1=1000, phase='val')

    print('Random Sample Generation for Training and Validation Finished')



# if __name__=="__main__":
#     Generate_Random_Samples(opt)
