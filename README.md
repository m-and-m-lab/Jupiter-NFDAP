
## Environment

Run the following to create the UNSB environment 

```
$ conda create -n UNSB python=3.8
$ conda activate UNSB
$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
$ conda install -c conda-forge packaging 
$ conda install -c "conda-forge/label/cf201901" visdom 
$ conda install -c conda-forge gputil 
$ conda install -c conda-forge dominate 
```

## Dataset Download
The Datasets used for the different test cases is stored in the folowing drive link. Download your preferred version and store it in the /dataset folder. Any custom dataset needs to have the trainA, trainB, testA and testB folders within them to run the module.

Drive link - https://drive.google.com/drive/u/0/folders/1sxrmw_-VGYsn_ybrHwWt4rLE82QSFpNN

## Training 
Run the following command to start training for a preferred dataset
```
python train.py --dataroot ./datasets/EZ_complete_data --name EZ_test \
--mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0
```

To run the training using the npy files (that contain I/F) and predict the additional UV and Methane channels:
```
python train.py --dataroot ./datasets/hst2jvh_GRS_npy --name hst2jvh_GRS_SB_npy \
--mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0 --dataset_mode unaligned_npy \
--phase train  --input_nc 5 --output_nc 5
```

Thre are several training parameters available in the codebase which you can change during training arguments. some of the important ones I have listed below - 
```
--dataset_mode
default='unaligned', chooses how datasets are loaded. [unaligned | aligned | single | colorization]

--serial_batches
if true, takes images in order to make batches, otherwise takes them randomly

--input_nc
default=3, # of input image channels: 3 for RGB and 1 for grayscale

--netD
default='basic_cond', choices=['basic', 'n_layers', 'pixel', 'patch', 'tilestylegan2', 'stylegan2'], specify discriminator architecture. The basic model is a 70x70 PatchGAN. n_layers specifies the layers in the discriminator

--netG
default='resnet_9blocks_cond', choices=['resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128', 'stylegan2', 'smallstylegan2', 'resnet_cat'], specify generator architecture
```

Some of the important train options are listed below - 
```
--n_epochs
default=200, number of epochs with the initial learning rate

--n_epochs_decay
default=200, number of epochs to linearly decay learning rate to zero

--gan_mode
default='lsgan', the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the CE objective
```

Although the training is available with arbitrary batch size, the recommended batch size is 1

## Test & Evaluation
Run the following command to test the model on the test dataset 

```
python test.py --dataroot [path-to-dataset] --name [experiment-name] --mode sb \
--phase test --epoch [epoch-for-test] --eval --num_test [num-test-image] \
--gpu_ids 0 --checkpoints_dir ./checkpoints
```

To run the testing using the npy files (that contain I/F) and save the predicted additional UV and Methane channels:
```
python test.py --dataroot ./datasets/hst2jvh_GRS_npy --name hst2jvh_GRS_SB_npy \
--checkpoints_dir ./checkpoints --mode sb --eval --phase test --gpu_ids 0 \
--dataset_mode unaligned_npy --input_nc 5 --output_nc 5
```

The outputs will be saved in ```./results/[experiment-name]/```

Folders named as ```fake_[num_NFE]``` represent the generated outputs with different NFE steps.

For evaluation, they use official module of [pytorch-fid](https://github.com/mseitzer/pytorch-fid)

```
python -m pytorch_fid [output-path] [real-path]
```

```real-path``` should be test images of target domain. 

For testing on vgg-based trained model, 

Refer the ```./vgg_sb/scripts/test_sc_main.sh``` file 

The pre-trained checkpoints are provided [here](https://drive.google.com/drive/folders/1Q8tuBGegMMHd9PzvcklDm0wM1sE4PPwK?usp=sharing)

