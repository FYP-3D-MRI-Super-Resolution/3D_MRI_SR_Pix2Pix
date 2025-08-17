import time
import torch
from torch.utils.data import DataLoader
from models import create_model
from util.visualizer import Visualizer
from util.util import print_timestamped
from data.MRI3DPatchDataset import MRI3DPatchDatasetPaired
from options.train_options import TrainOptions

from options.train_options import TrainOptions

opt = TrainOptions().parse()

opt.model = 'pix2pix3d'
opt.dataset_mode = 'MRI3DPatchDataset'
opt.input_nc = 1
opt.output_nc = 1
opt.ngf = 64
opt.netG = 'unet_32'
opt.threed = True
opt.gpu_ids = [0]
opt.use_dropout = False
opt.init_type = 'normal'
opt.init_gain = 0.02
opt.batch_size = 2
opt.epoch_count = 1
opt.n_epochs = 100
opt.n_epochs_decay = 0
opt.display_id = -1
opt.print_model_info = False
opt.save_latest_freq = 500
opt.save_by_iter = False
opt.save_epoch_freq = 5
opt.display_freq = 500
opt.print_freq = 100
opt.update_html_freq = 500
opt.crop_size = 32
opt.load_size = 32
opt.preprocess = 'none'


low_res_dir = "/kaggle/input/high-res-and-low-res-mri/Refined-MRI-dataset/Low-Res"
high_res_dir = "/kaggle/input/high-res-and-low-res-mri/Refined-MRI-dataset/High-Res"
patch_size = (32, 32, 32)

def create_dataset(opt):
    dataset = MRI3DPatchDatasetPaired(low_res_dir, high_res_dir, patch_size)
    return dataset

dataset = create_dataset(opt)
dataset_size = len(dataset)
print('The number of training images = %d' % dataset_size)

model = create_model(opt)
model.setup(opt)

if opt.print_model_info and "pix2pix" in opt.model:
    from torchsummary import summary
    final_size = opt.crop_size if "crop" in opt.preprocess else opt.load_size
    if "3d" in opt.model:
        summary(model.netG, (opt.input_nc, final_size, final_size, final_size))
    else:
        summary(model.netG, (opt.input_nc, final_size, final_size))

visualizer = Visualizer(opt)

total_iters = 0
init_time = time.time()
for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
    epoch_start_time = time.time()
    iter_data_time = time.time()
    epoch_iter = 0
    visualizer.reset()

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += opt.batch_size
        epoch_iter += opt.batch_size

        # set input and optimize
        model.set_input(data)
        model.optimize_parameters()

        # display and save images
        if total_iters % opt.display_freq == 0:
            save_result = total_iters % opt.update_html_freq == 0
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        # print losses
        if total_iters % opt.print_freq == 0:
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            if opt.display_id > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

        # save latest checkpoint
        if total_iters % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)

        iter_data_time = time.time()

    model.update_learning_rate()

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

end_time = round(time.time() - init_time, 3)
print_timestamped("The training process took " + str(end_time) + "s.")
