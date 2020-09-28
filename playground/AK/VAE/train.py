import argparse
import torch
import torch.optim as optim
from torchvision.utils import make_grid
from torch.utils.data.sampler import SubsetRandomSampler

import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from genEM3.model.VAE import ConvVAE
from genEM3.data.wkwdata import WkwData, DataSplit
import genEM3.util.path as gpath
from genEM3.util.image import undo_normalize
from genEM3.util import gpu
from genEM3.data.transforms.normalize import ToStandardNormal
from genEM3.util.tensorboard import launch_tb, add_graph
from genEM3.training.VAE import train, test, save_checkpoint

# set the proper device (GPU with a specific ID or cpu)
cuda = False
gpu_id = 1
if cuda:
    print(f'Using GPU: {gpu_id}')
    gpu.get_gpu(gpu_id)
    device = torch.device(torch.cuda.current_device())
else:
    device = torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description='Convolutional VAE MNIST Example')
    parser.add_argument('--result_dir', type=str, default='.log', metavar='DIR',
                        help='output directory')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: None')

    # model options
    # Note(AK): with the AE models from genEM3, the 2048 latent size and 16 fmaps are fixed
    parser.add_argument('--latent_size', type=int, default=2048, metavar='N',
                        help='latent vector size of encoder')
    parser.add_argument('--weight_KLD', type=float, default=1.0, metavar='N',
                        help='Weight for the KLD part of loss')

    args = parser.parse_args()
    print('The command line argument:\n')
    print(args)

    # Make the directory for the result output
    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)

    torch.manual_seed(args.seed)
    # Parameters
    connDataDir = '/conndata/alik/genEM3_runs/VAE/'
    json_dir = gpath.getDataDir()
    datasources_json_path = os.path.join(json_dir, 'datasource_20X_980_980_1000bboxes.json')
    input_shape = (140, 140, 1)
    output_shape = (140, 140, 1)
    data_sources = WkwData.datasources_from_json(datasources_json_path)
    # Only pick the first bboxes for faster epoch
    data_sources = [data_sources[0]]
    data_split = DataSplit(train=0.80, validation=0.00, test=0.20)
    cache_RAM = True
    cache_HDD = True
    cache_root = os.path.join(connDataDir, '.cache/')
    gpath.mkdir(cache_root)

    # Set up summary writer for tensorboard
    constructedDirName = ''.join([f'weightedVAE_{args.weight_KLD}_', gpath.gethostnameTimeString()])
    tensorBoardDir = os.path.join(connDataDir, constructedDirName)
    writer = SummaryWriter(log_dir=tensorBoardDir)
    launch_tb(logdir=tensorBoardDir, port='7900')
    # Set up data loaders
    num_workers = 8
    dataset = WkwData(
        input_shape=input_shape,
        target_shape=output_shape,
        data_sources=data_sources,
        data_split=data_split,
        normalize=False,
        transforms=ToStandardNormal(mean=148.0, std=36.0),
        cache_RAM=cache_RAM,
        cache_HDD=cache_HDD,
        cache_HDD_root=cache_root
    )
    # Data loaders for training and test
    train_sampler = SubsetRandomSampler(dataset.data_train_inds)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler,
        collate_fn=dataset.collate_fn)

    test_sampler = SubsetRandomSampler(dataset.data_test_inds)
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=args.batch_size, num_workers=num_workers, sampler=test_sampler,
        collate_fn=dataset.collate_fn)
    # Model and optimizer definition
    input_size = 140
    output_size = 140
    kernel_size = 3
    stride = 1
    weight_KLD = args.weight_KLD
    model = ConvVAE(latent_size=args.latent_size,
                    input_size=input_size,
                    output_size=output_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    weight_KLD=weight_KLD).to(device)
    # Add model to the tensorboard as graph
    add_graph(writer=writer, model=model, data_loader=train_loader, device=device)
    # print the details of the model
    print_model = True
    if print_model:
        model.summary(input_size=input_size, device=device.type)
    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    start_epoch = 0
    best_test_loss = np.finfo('f').max

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint %s' % args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_test_loss = checkpoint['best_test_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint %s' % args.resume)
        else:
            print('=> no checkpoint found at %s' % args.resume)
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_lossDetailed = train(epoch, model, train_loader, optimizer, args,
                                               device=device)
        test_loss, test_lossDetailed = test(epoch, model, test_loader, writer, args,
                                            device=device)

        # logging
        writer.add_scalar('loss_train/total', train_loss, epoch)
        writer.add_scalar('loss_test/total', test_loss, epoch)
        writer.add_scalars('loss_train', train_lossDetailed, global_step=epoch)
        writer.add_scalars('loss_test', test_lossDetailed, global_step=epoch)
        # add the histogram of weights and biases plus their gradients
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.detach().cpu().data.numpy(), epoch)
            # weight_KLD is a parameter but does not have a gradient. It creates an error if one 
            # tries to plot the histogram of a None variable
            if param.grad is not None:
                writer.add_histogram(name+'_gradient', param.grad.cpu().numpy(), epoch)
        # plot mu and logvar
        for latent_prop in ['cur_mu', 'cur_logvar']:
            latent_val = getattr(model, latent_prop)
            writer.add_histogram(latent_prop, latent_val.cpu().numpy(), epoch)
        # flush them to the output
        writer.flush()
        print('Epoch [%d/%d] loss: %.3f val_loss: %.3f' % (epoch + 1, args.epochs, train_loss, test_loss))
        is_best = test_loss < best_test_loss
        best_test_loss = min(test_loss, best_test_loss)
        save_directory = os.path.join(tensorBoardDir, '.log')
        save_checkpoint({'epoch': epoch,
                         'best_test_loss': best_test_loss,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        is_best,
                        save_directory)

        with torch.no_grad():
            # Image 64 random sample from the prior latent space and decode
            sample = torch.randn(64, args.latent_size).to(device)
            sample = model.decode(sample).cpu()
            sample_uint8 = undo_normalize(sample, mean=148.0, std=36.0)
            img = make_grid(sample_uint8)
            writer.add_image('sampling', img, epoch)


# Run main as script
if __name__ == '__main__':
    main()
