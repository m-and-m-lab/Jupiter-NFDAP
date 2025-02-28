python test.py --dataroot ./datasets/horse2zebra --name h2z_SB --checkpoints_dir ./checkpoints --mode sb --eval --phase test --num_test 120 --gpu_ids 0


python test.py --dataroot ./datasets/hst2jvh_GRS_npy --name hst2jvh_GRS_SB_npy --checkpoints_dir ./checkpoints --mode sb --eval --phase test --num_test 8 --gpu_ids 0 --dataset_mode unaligned_npy --input_nc 5 --output_nc 5
