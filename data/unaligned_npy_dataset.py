import os.path
from data.base_dataset import BaseDataset, get_transform, normalize_npy
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import numpy as np
import torch
import math


class UnalignedNPYDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets specifically for npy files
    Edited to accomodate the npy loading and preprocessing of 5 channels for img B


    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        ## Filter A and B based on given zones
        if len(self.opt.zones)>0: # if no zones are given then don't filter
            self.A_paths = self.filter_data_by_zones(self.A_paths)
            self.B_paths = self.filter_data_by_zones(self.B_paths)

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        ## If input channels is 5, sample fake UV and methane data for JunoCam (A) to account for the channel number difference to HST (B)
        ## For now a truncated normal distribution between (-1, 1)
        ## These should be sampled only once at the beginning of training for all A training images
        # ** Not sure how this affects the losses -- PatchNCE only uses within negative patches so it should be unaffected
        if self.opt.input_nc == 5:
            tensor_empty = torch.zeros((self.A_size, 2, self.opt.crop_size, self.opt.crop_size))
            self.UV_methane_random = self.trunc_normal(tensor_empty, mean=0, std=1, a=-1, b=1)
        #UV_methane_random = torch.randn(2, self.opt.crop_size, self.opt.crop_size)
        #UV_methane_random = torch.normal(mean=0, std=0.5, size=(2, self.opt.crop_size, self.opt.crop_size))


        # ** Add the architecture of the denseNorm for the generator
        # ** Add the physics-informed loss
        # ** Check whether it makes sense to add a cycleGAN objective


    def filter_data_by_zones(self, paths):
        filtered_paths = []
        for i in range(len(paths)):
            zone_path = paths[i].split('/')[-1].split('_')[0]
            #print(zone_path)
            if zone_path in self.opt.zones:
                filtered_paths.append(paths[i]) 
        return filtered_paths


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        index_A = index % self.A_size
        A_path = self.A_paths[index_A]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        ## ** Hardcode the path for now
        #A_path = "./datasets/junocam_calibration_GRS_TINY_npy/trainA/GRS_PJ_18_jc_000098.npy"
        #B_path = "./datasets/junocam_calibration_GRS_TINY_npy/trainB/GRS_cycle_23_rot_B_hst_000166.npy"

        # npy data are already normalized between 0...1        
        A_img = np.load(A_path)
        B_img = np.load(B_path)

        A_img = torch.tensor(A_img)
        A_img = torch.permute(A_img, (2, 0, 1)) # n_channels x H x W
        B_img = torch.tensor(B_img) # already in: n_channels x H x W

        ## Do correction for the Methane band
        # GRS is supposed to have the highest values in Methane band and its values are between 0.12-0.18
        # So probably a 5.0 multiplier is fine
        B_img[4,:,:] = B_img[4,:,:] * 5.0
        # import matplotlib.pyplot as plt
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
        # ax1.imshow(B_img[0,:,:], aspect='equal', cmap='gray')
        # ax1.set_title("UV")
        # ax2.imshow(B_img[4,:,:], aspect='equal', cmap='gray')
        # ax2.set_title("Methane")
        # plt.show()

        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        
        # convert=False because inputs are already tensors and we use a later function to normalize
        transform = get_transform(modified_opt, convert=False)
        A = transform(A_img)
        B = transform(B_img)

        # Custom function to normalize inputs of arbitrary number of channels to [-1, 1]
        A = normalize_npy(A) # 3 x H x W -- B, G, R
        B = normalize_npy(B) # 5 x H x W -- UV, B, G, R, Methane 

        # Order channels of B to match the ones from A: (B, G, R, UV, Methane) -- regardless of the input_nc option
        order = torch.tensor([1, 2, 3, 0, 4])
        B = B[order, :, :]

        if self.opt.input_nc == 5:
            # Append the fake UV, Methane to A
            A = torch.cat((A, self.UV_methane_random[index_A, :, :, :]), dim=0) # 5 x H x W
        
        elif self.opt.input_nc == 3:
            # Remove UV, Methane from HST
            B = B[:3,:,:]

        elif self.opt.input_nc == 2:
            # This is used to predict UV, Methane from JunoCam color (during a two-stage prediction strategy)
            # Keep only UV, Methane from HST
            B_orig = B.clone()
            B = B[3:,:,:]
            A_orig = A.clone()
            # Use R, B channels for predicting UV,Methane
            A = A[torch.tensor([0,2]),:,:]
            # Randomly choose which JunoCam color channels to include in a 2-channel input of A
            #idx = torch.randperm(3)
            #A = A[idx,:,:]
            #A = A[:2,:,:]
        else:
            raise Exception('Chosen input number of channels not supported for this dataset!')


        if self.opt.input_nc == 2:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_orig': A_orig, 'B_orig': B_orig}

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


    def trunc_normal(self, tensor, mean, std, a, b):
        # Returns a tensor filled with a truncated normal distribution
        # Source: https://github.com/pytorch/pytorch/blob/a40812de534b42fcf0eb57a5cecbfdc7a70100cf/torch/nn/init.py#L22
        # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        def norm_cdf(x):
            # Computes standard normal cumulative distribution function
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        if (mean < a - 2 * std) or (mean > b + 2 * std):
            warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                        "The distribution of values may be incorrect.",
                        stacklevel=2)

        with torch.no_grad():
            # Values are generated by using a truncated uniform distribution and
            # then using the inverse CDF for the normal distribution.
            # Get upper and lower cdf values
            l = norm_cdf((a - mean) / std)
            u = norm_cdf((b - mean) / std)

            # Uniformly fill tensor with values from [l, u], then translate to
            # [2l-1, 2u-1].
            tensor.uniform_(2 * l - 1, 2 * u - 1)

            # Use inverse cdf transform for normal distribution to get truncated
            # standard normal
            tensor.erfinv_()

            # Transform to proper mean, std
            tensor.mul_(std * math.sqrt(2.))
            tensor.add_(mean)

            # Clamp to ensure it's in the proper range
            tensor.clamp_(min=a, max=b)
            return tensor