import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer


def get_visuals(opt, model, data):
    # Returns inputs+predictions for visualization
    visuals = model.get_current_visuals()

    if opt.dataset_mode == "unaligned_npy":
        # Organize visuals to separate RGB from 5 channels and include UV and methane
        # When input_nc is 2 then use the original A image (fake Juno) to display next to predicted UV, Methane
        new_visuals = {}
        if opt.input_nc==2:
            new_visuals['A_calibrated_JVH'] = torch.flip(data['A_orig'], dims=[1]) # converting BGR to RGB
            new_visuals['B_HST'] = torch.flip(data['B_orig'][:,:3,:,:], dims=[1]) # HST RGB
            new_visuals['B_HST_UV'] = data['B_orig'][:,3,:,:].unsqueeze(0)
            new_visuals['B_HST_M'] = data['B_orig'][:,4,:,:].unsqueeze(0)
            new_visuals['fake_UV'] = visuals['fake_B'][:,0,:,:].unsqueeze(0)
            new_visuals['fake_M'] = visuals['fake_B'][:,1,:,:].unsqueeze(0)
        else:
            for k,v in visuals.items():
                BGR = v[:, :3, :, :]
                new_visuals[k] = torch.flip(BGR, dims=[1]) # converting BGR to RGB
            if opt.input_nc == 5:
                new_visuals['fake_B_UV'] = visuals['fake_B'][:, 3, :, :].unsqueeze(0) # Predicted UV
                new_visuals['fake_B_M'] = visuals['fake_B'][:, 4, :, :].unsqueeze(0) # Predicted Methane
                new_visuals['real_B_UV'] = visuals['real_B'][:, 3, :, :].unsqueeze(0)
                new_visuals['real_B_M'] = visuals['real_B'][:, 4, :, :].unsqueeze(0)
        return new_visuals
    else:
        return visuals


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset2 = create_dataset(opt)
    dataset_size = len(dataset)    # get the number of images in the dataset.

    #print(dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        for i, (data,data2) in enumerate(zip(dataset,dataset2)):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data,data2)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data,data2)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(get_visuals(opt, model, data), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
