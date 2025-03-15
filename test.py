"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import torch
import numpy as np


def get_new_visuals(visuals):
    # Returns visuals only for npy dataset
    # Organize visuals to separate RGB from 5 channels and include UV and methane
    new_visuals = {}
    if opt.input_nc == 2:
        new_visuals['A_calibrated_JVH'] = torch.flip(data['A_orig'], dims=[1]) # converting BGR to RGB
        new_visuals['fake_5_UV'] = visuals['fake_5'][:,0,:,:].unsqueeze(0)
        new_visuals['fake_5_M'] = visuals['fake_5'][:,1,:,:].unsqueeze(0)
    else:
        for k,v in visuals.items():
            BGR = v[:, :3, :, :]
            new_visuals[k] = torch.flip(BGR, dims=[1]) # converting BGR to RGB
        if opt.input_nc == 5:
            new_visuals['fake_5_UV'] = visuals['fake_5'][:, 3, :, :].unsqueeze(0) # Predicted UV
            new_visuals['fake_5_M'] = visuals['fake_5'][:, 4, :, :].unsqueeze(0) # Predicted Methane
    return new_visuals


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset2 = create_dataset(opt)
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    for i, (data,data2) in enumerate(zip(dataset,dataset2)):
        if i == 0:
            model.data_dependent_initialize(data,data2)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data,data2)  # unpack data from data loader
        model.test()           # run inference
        
        visuals = model.get_current_visuals()  # get image results
        #print(visuals.keys()) # ['real', 'fake_1', 'fake_2', 'fake_3', 'fake_4', 'fake_5']
        
        if opt.dataset_mode == "unaligned_npy":
            
            new_visuals = get_new_visuals(visuals)

            img_path = model.get_image_paths() # get image paths

            # Save npy file
            #print(web_dir)
            save_path = web_dir + "/npys/fake_5/" #+ img_path[0] +
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            npy_save_file = save_path + img_path[0].split('/')[-1]
            fake_5 = util.tensor2im(visuals['fake_5'], imtype=np.float32) / 255.0
            np.save(npy_save_file, fake_5) # H x W x C -- B, G, R, UV, Methane

            # change path to png from npy
            part1 = os.path.join(*img_path[0].split('/')[:-1])
            part2 = img_path[0].split('/')[-1].split('.')[0] + ".png"
            img_path = [part1+"/"+part2]

            save_images(webpage, new_visuals, img_path, width=opt.display_winsize)
        else:
            save_images(webpage, visuals, img_path, width=opt.display_winsize)
        
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        
    webpage.save()  # save the HTML
