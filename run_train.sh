#python train.py --dataroot ./datasets/horse2zebra --name h2z_SB --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 1


#python train.py --dataroot ./datasets/hst2jvh_GRS --name hst2jvh_GRS_SB_0 --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0 --phase train

python train.py --dataroot ./datasets/hst2jvh_GRS --name hst2jvh_GRS_SB --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0 --dataset_mode unaligned --phase train --no_html


## Run training using the npy files to predict UV and methane
python train.py --dataroot ./datasets/hst2jvh_GRS_npy --name hst2jvh_GRS_SB_npy --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0 --dataset_mode unaligned_npy --phase train  --input_nc 5 --output_nc 5


# When debugging is useful to use:
--num_threads 1

# Use this to take images in order to make batches -- we have much more HST segments than JVH so this would still not align the data based on zone
--serial_batches

