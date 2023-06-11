"""
This function is used to generate predicted probability for testing images
Input: path of testing image
Output: Correspondig probability images and 2D mip images for visualization under the same whole path
"""

from Common import *
from utils.tools import *
import tifffile
import torch.nn as nn
import time
import numpy as np
from dataset.generatePatches import *
from torch.utils.data import DataLoader
import skimage.io as skio
from glob import glob
import unicodecsv
from config import opt


all_mean1 = np.array([168.5], dtype=np.float32)
all_std1 = np.array([500], dtype=np.float32)

"""This is only used to generate mip images for visualization"""
def ImageForVis(image_recon):
    """Just for Visualization"""
    image_recon = np.array(image_recon, np.int32)
    image_recon[image_recon > 350] = 350
    image_recon[image_recon < 100] = 100
    image_recon = (image_recon - np.min(image_recon)) / (np.max(image_recon) - np.min(image_recon))
    image_recon=np.uint8(image_recon * 255)
    return image_recon

"""DataSet Loader for new testing images"""
class GenerateDataset_ForNew():
    def __init__(self,image,image_shape, patch_size, overlap):
        self.n_patches = np.ceil(image_shape / (patch_size - overlap))
        self.patches_num = int(self.n_patches[0]*self.n_patches[1]*self.n_patches[2])
        self.patch_size=patch_size

        self.image=image.astype(np.float32)
        self.image=np.clip(self.image,opt.limited_range[0],opt.limited_range[1])
        self.image=(self.image-all_mean1)/all_std1

        # calculate patch index
        self.patchindices = compute_patch_indices(image_shape, patch_size, overlap,start=0)

    def __len__(self):
        return self.patches_num

    def __getitem__(self, ind ):
        # get the patches of the image and label
        image_patch = get_patch_from_3d_data(self.image, patch_shape=self.patch_size, patch_index=self.patchindices[ind, :])

        # To expand the dim of the dataset and turn the dataset into torch
        image_patch=np.expand_dims(image_patch,axis=0)
        image_patch=torch.from_numpy(image_patch).float()

        return image_patch


def test_model(val_loader,model):
    model.eval()

    pred_Patches = []
    prob_patches =[]
    soft_max = nn.Softmax(dim=1)

    for batch_ids, (image_patch) in enumerate(val_loader):
        if opt.use_cuda:
            image_patch=image_patch.cuda()
            output=model(image_patch)

            with torch.no_grad():
                # just 0 and 1
                _,pred_patch=torch.max(output,dim=1)

                # for prob
                prob_patch = soft_max(output)
                prob_patch=prob_patch[:,1,...]
                del output

                pred_patch=pred_patch.cpu().numpy()
                prob_patch = prob_patch.cpu().numpy()

                for id1 in range(pred_patch.shape[0]):
                    pred1=np.array(pred_patch[id1,:,:,:], dtype=np.float32)
                    pred_Patches.append( pred1 )

                    prob1 = np.array(prob_patch[id1, :, :, :], dtype=np.float32)
                    prob_patches.append(prob1)


    return pred_Patches,prob_patches




if __name__=="__main__":
    ##############################################################################################
    # 1. get the image path and corrspodning probability path and 2D mip path
    image_path=opt.test_paths+'/image'

    prob_path=opt.test_paths+'/'+opt.dataset_prefix+'_prob'
    mip_path=opt.test_paths+'/'+opt.dataset_prefix+'_mip'

    remake_dirs(prob_path)
    remake_dirs(mip_path)

    ##############################################################################################
    # 2. load the model and parameter
    model = GetModel(opt)
    parameters_name = 'checkpoints/DHPR_300.ckpt'
    model_CKPT = torch.load(parameters_name)
    model.load_state_dict(model_CKPT['state_dict'])
    print('loading checkpoint!')

    ##############################################################################################
    # 3. begin to test the images
    file_names = glob(os.path.join(image_path, '*.tif'))

    for image_num in range(len(file_names)):
        file_name1 = os.path.join(image_path,file_names[image_num])
        save_num=file_name1.split('/')[-1].split('.')[0]

        # read the image and process
        start1=time.time()
        image=tifffile.imread(file_name1)
        image=np.array(image,np.uint16)

        # begin to generate patches
        patch_size=opt.test_patch_size
        overlap=opt.test_overlap
        patch_indices=compute_patch_indices(image.shape, patch_size, overlap, start=0)

        Tdataset = GenerateDataset_ForNew(image, image.shape, patch_size, overlap)
        val_loader = DataLoader(Tdataset, batch_size=4, num_workers=5, shuffle=False)

        pred_Patches,prob_patches=test_model(val_loader, model)

        # begin to combine the images into one
        patchindices = compute_patch_indices(image.shape, patch_size, overlap,start=0)
        prob_recon = reconstruct_from_patches(prob_patches, patchindices, image.shape)

        run_time=time.time()-start1
        print(run_time)

        tifffile.imsave(os.path.join(prob_path, save_num + '_prob.tif'), np.uint8(prob_recon * 255))

        im_mip=np.max(ImageForVis(image),0)
        im_mip = im_mip.astype(np.uint8)

        prob_mip=np.max(np.uint8(prob_recon * 255),0)
        prob_mip=prob_mip.astype(np.uint8)

        skio.imsave(mip_path + '/' + save_num + '_mip.png', np.vstack([im_mip,prob_mip]))


        print('ok')

































