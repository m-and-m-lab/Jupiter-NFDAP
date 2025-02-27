#python train.py --dataroot ./datasets/horse2zebra --name h2z_SB --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 1


#python train.py --dataroot ./datasets/hst2jvh_GRS --name hst2jvh_GRS_SB_0 --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0 --phase train

python train.py --dataroot ./datasets/hst2jvh_GRS --name hst2jvh_GRS_SB_0 --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0 --dataset_mode unaligned --phase train --no_html --num_threads 1 --serial_batches


## to run through the new npy dataloader
python train.py --dataroot ./datasets/hst2jvh_GRS_npy --name hst2jvh_GRS_SB_0 --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0 --dataset_mode unaligned_npy --phase train --no_html --num_threads 1 --serial_batches
