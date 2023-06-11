"""
This function is used to find representative samples that have high diversity
and hardness from the input images using DHPR selection rule

Input: opt.all_paths--- all paths that include the training image and label
                    images are saved in the "image" file under opt.all_paths file
                    labels are saved in the "label" file under opt.all_paths file
                    corresponding images and labels should share the same names
Output: selected index of the selected images for training
"""

import os
import tifffile
import numpy as np
from config import opt

def ROI_Range_WithLimit(xx1,shape1,limit1=200):
    x_min=np.min(xx1)
    x_max=np.max(xx1)

    if x_max-x_min+1<limit1:
        radius1 = limit1 / 2
        center1=(x_min+x_max)/2
        x_min=np.max([0,center1-radius1])
        x_max=np.min([shape1,center1+radius1])

    x_min=int(x_min)
    x_max=int(x_max)
    return x_min,x_max

def ROI_Region_WithLimit(image1,label1,limit1=200):
    """crop the image to find the interested area, keep a min range to estimate the background"""
    xx1,yy1,zz1=np.where(label1>0)
    x_min,x_max=ROI_Range_WithLimit(xx1,image1.shape[0],limit1)
    y_min, y_max = ROI_Range_WithLimit(yy1,image1.shape[1],limit1)
    z_min, z_max = ROI_Range_WithLimit(zz1,image1.shape[2],limit1)

    crange=[x_min,x_max,y_min, y_max,z_min, z_max]
    image2=image1[x_min:x_max,y_min:y_max,z_min:z_max]
    label2=label1[x_min:x_max,y_min:y_max,z_min:z_max]
    return image2,label2,crange

def calculate_various_criterions(image1,label1):
    image1, label1, crange=ROI_Region_WithLimit(image1,label1,200)

    # 1. calculate the signal density
    vox_nums = np.sum(label1)
    signal_density=vox_nums/50000
    signal_density=np.clip(signal_density,0,1)

    # 2. calculate the noise variance
    background=image1[label1==0].flatten()
    background = np.random.permutation(background)[0:80000]
    background_mean=np.mean(background)
    # remove confused voxels for background calculation
    index_temp = np.where(background >background_mean+60)
    background=np.delete(background,index_temp)
    background_mean = np.mean(background)
    background_std = np.std(background)
    noise_variance=background_std

    # 3. calculate the low snr ratio
    foreground = image1[label1 > 0].flatten()

    diff_value=np.array(foreground,np.float)-np.array(background_mean,np.float)
    # restriction 1 to avoid large bias
    background_std1 = np.min([background_std, 6.5])

    snr_value = diff_value / np.array(background_std1 + 1e-5)/2
    snr_value = snr_value.flatten()

    lsnr_ratio=len(np.array(np.where(snr_value<2)).flatten())/len(snr_value)

    return signal_density,noise_variance,lsnr_ratio


def selected_rule1(info_all,index1,top_num=300):
    r_diversity=info_all[index1, 0]
    r_lsnr=info_all[index1, 2]

    r_all=r_diversity+r_lsnr
    index1_sort=np.argsort(-r_all)

    index1_selected=index1[index1_sort[:top_num]]
    index1_selected =np.sort(index1_selected)

    return index1_selected


if __name__=="__main__":
    ####################################################################
    # 1. input the image path and label path
    all_paths=opt.all_paths
    image_path = os.path.join(all_paths,'image')
    label_path = os.path.join(all_paths,'label')

    file_names = os.listdir(image_path)
    all_num=len(file_names)

    ####################################################################
    # 2. begin to calculate the information
    info_all=np.zeros([all_num,3])

    for num1 in range(all_num):
        image_name=image_path+'/{}'.format(file_names[num1])
        label_name=label_path+'/{}'.format(file_names[num1])

        ###################### begin to calculate the information
        image1=tifffile.imread(image_name)
        label1=tifffile.imread(label_name)
        label1=label1>0

        signal_density, noise_variance, lsnr_ratio=calculate_various_criterions(image1,label1)

        info_all[num1,0]=signal_density
        info_all[num1, 1] = noise_variance
        info_all[num1, 2] = lsnr_ratio

        print(num1,signal_density, noise_variance, lsnr_ratio)

    ####################################################################
    # 3. save the calculated information
    info_name=all_paths+'/sample_information.npy'
    np.save(info_name, info_all)

    ####################################################################
    # 4. search out images with high diversity and hardness
    # calculate the median value of noise variance
    median_noise_variance=np.median(info_all[:,1])
    index1 = np.where(info_all[:,1]>median_noise_variance)
    index1=np.array(index1).flatten()
    print(len(index1))

    index1_selected =selected_rule1(info_all, index1, np.min([300,len(index1)]))
    index1_selected=np.sort(index1_selected)

    sind1_name=all_paths+'/selectd_rule1.npy'
    np.save(sind1_name,index1_selected)

    print('Infomration Calculation Finished')



