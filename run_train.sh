#python train.py --dataroot ./datasets/horse2zebra --name h2z_SB --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 1


#python train.py --dataroot ./datasets/hst2jvh_GRS --name hst2jvh_GRS_SB_0 --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0 --phase train

python train.py --dataroot ./datasets/hst2jvh_GRS --name hst2jvh_GRS_SB --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0 --dataset_mode unaligned --phase train --no_html


## Run training using the npy files to predict UV and methane
python train.py --dataroot ./datasets/hst2jvh_GRS_npy --name hst2jvh_GRS_SB_npy --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0 --dataset_mode unaligned_npy --phase train  --input_nc 5 --output_nc 5


### Two-stage training (example with hardcoded single training example)

# 1. Run training of TINY GRS with npy files for RGB to RGB prediction
python train.py --dataroot ./datasets/junocam_calibration_GRS_TINY_npy --name junocam_calibration_GRS_TINY_npy_SB_single --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0 --dataset_mode unaligned_npy --phase train --input_nc 3 --output_nc 3

# 2. Run testing on the training data to produce the RGB predictions (assume testA contains same data as trainA)
python test.py --dataroot ./datasets/junocam_calibration_GRS_TINY_npy --name junocam_calibration_GRS_TINY_npy_SB_single --checkpoints_dir ./checkpoints --mode sb --eval --phase test --num_test 1 --gpu_ids 0 --dataset_mode unaligned_npy --input_nc 3 --output_nc 3

# 3. Create a new folder (junocam_calibration_GRS_TINY_npy_stage2) where the fake RGB junocam is domain A (source) and the HST is domain B target 

# 4. Run training using stage2 data for fake RGB to UV,Methane prediction
python train.py --dataroot ./datasets/junocam_calibration_GRS_TINY_npy_stage2 --name junocam_calibration_GRS_TINY_npy_SB_single_UVM --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0 --dataset_mode unaligned_npy --phase train --input_nc 2 --output_nc 2


# When debugging is useful to use:
--num_threads 1

# Use this to take images in order to make batches -- we have much more HST segments than JVH so this would still not align the data based on zone
--serial_batches

