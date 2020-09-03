import argparse
import torch
import torch.optim as optim
from torch.nn import functional as F
from torchvision.utils import make_grid
from torch.utils.data.sampler import SubsetRandomSampler

import os
import shutil
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

from genEM3.model.VAE import ConvVAE
from genEM3.data.wkwdata import WkwData, DataSplit
import genEM3.util.path as gpath
from genEM3.util.image import undo_normalize
from genEM3.util import gpu
from genEM3.data.transforms.normalize import ToStandardNormal

# factor for numerical stabilization of the loss sum
NUMFACTOR = 10000

# set the proper device (GPU with a specific ID or cpu)
cuda = True
gpu_id = 1
if cuda:
    print(f'Using GPU: {gpu_id}')
    gpu.get_gpu(gpu_id)
    device = torch.device(torch.cuda.current_device())
else:
    device = torch.device("cpu")


def loss_function(recon_x, x, mu, logvar):
    img_size_recon = torch.tensor(recon_x.shape[2:4]).prod()
    img_size_input = torch.tensor(x.shape[2:4]).prod()
    # reconstruction loss
    BCE = F.mse_loss(recon_x.view(-1, img_size_recon), x.view(-1, img_size_input), reduction='sum')
    # KL divergence loss between the posterior and prior of latent space
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Add a dict of separate reconstruction and KL loss
    lossDetail = {'Recon': BCE, 'KLD': KLD}
    return BCE + KLD, lossDetail


def train(epoch, model, train_loader, optimizer, args):
    model.train()
    train_loss = 0
    detailedLoss = {'Recon': 0.0, 'KLD': 0.0}
    for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='train'):
        data = data['input'].to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)

        loss, curDetLoss = loss_function(recon_batch, data, mu, logvar)
        train_loss += (loss.item() / NUMFACTOR)
        # Separate loss
        for key in curDetLoss:
            detailedLoss[key] += (curDetLoss.get(key) / NUMFACTOR)
        # Backprop
        loss.backward()
        optimizer.step()
    num_data_points = len(train_loader.dataset)
    train_loss /= num_data_points
    train_loss *= NUMFACTOR

    for key in detailedLoss:
        detailedLoss[key] /= num_data_points
        detailedLoss[key] *= NUMFACTOR

    return train_loss, detailedLoss


def test(epoch, model, test_loader, writer, args):
    model.eval()
    test_loss = 0
    detailedLoss = {'Recon': 0.0, 'KLD': 0.0}
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader), total=len(test_loader), desc='test'):
            data = data['input'].to(device)
            recon_batch, mu, logvar = model(data)
            curLoss, curDetLoss = loss_function(recon_batch, data, mu, logvar)
            test_loss += (curLoss.item() / NUMFACTOR)
            # The separate KL and Reconstruction losses
            for key in curDetLoss:
                detailedLoss[key] += (curDetLoss.get(key) / NUMFACTOR)
            # Add 8 test images and reconstructions to tensorboard
            if batch_idx == 0:
                n = min(data.size(0), 8)
                # concatenate the input data and associated reconstruction
                comparison = torch.cat([data[:n], recon_batch[:n]]).cpu()
                comparison_uint8 = undo_normalize(comparison, mean=148.0, std=36.0)
                img = make_grid(comparison_uint8)
                writer.add_image('test_reconstruction',
                                 img,
                                 epoch)
                # save_image(comparison.cpu(), 'results/reconstruction_' + str(epoch) + '.png', nrow=n)
    # Divide by the length of the dataset and multiply by factor used for numerical stabilization
    num_data_points = len(test_loader.dataset)
    test_loss /= num_data_points
    test_loss *= NUMFACTOR

    for key in detailedLoss:
        detailedLoss[key] /= num_data_points
        detailedLoss[key] *= NUMFACTOR

    return test_loss, detailedLoss


def save_checkpoint(state, is_best, outdir='.log'):
    checkpoint_file = os.path.join(outdir, 'checkpoint.pth')
    best_file = os.path.join(outdir, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)


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

    args = parser.parse_args()
    print('The command line argument:\n')
    print(args)

    # Make the directory for the result output
    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)

    torch.manual_seed(args.seed)

#     kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    # Parameters
    connDataDir = '/conndata/alik/genEM3_runs/VAE/'
    json_dir = gpath.getDataDir()
    datasources_json_path = os.path.join(json_dir, 'datasource_20X_980_980_1000bboxes.json')
    input_shape = (140, 140, 1)
    output_shape = (140, 140, 1)
    data_sources = WkwData.datasources_from_json(datasources_json_path)
    # Only pick the first two bboxes for faster epoch
    data_split = DataSplit(train=0.80, validation=0.00, test=0.20)
    cache_RAM = True
    cache_HDD = True
    cache_root = os.path.join(connDataDir, '.cache/')
    gpath.mkdir(cache_root)

    num_workers = 8
    data_sources = [data_sources[0]]
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
    model = ConvVAE(latent_size=args.latent_size,
                    input_size=input_size,
                    output_size=output_size,
                    kernel_size=kernel_size,
                    stride=stride).to(device)
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
    tensorBoardDir = os.path.join(connDataDir, gpath.gethostnameTimeString())
    writer = SummaryWriter(logdir=tensorBoardDir)

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_lossDetailed = train(epoch, model, train_loader, optimizer, args)
        test_loss, test_lossDetailed = test(epoch, model, test_loader, writer, args)

        # logging
        writer.add_scalar('loss_train/total', train_loss, epoch)
        writer.add_scalar('loss_test/total', test_loss, epoch)
        writer.add_scalars('loss_train', train_lossDetailed, global_step=epoch)
        writer.add_scalars('loss_test', test_lossDetailed, global_step=epoch)
        print('Epoch [%d/%d] loss: %.3f val_loss: %.3f' % (epoch + 1, args.epochs, train_loss, test_loss))

        is_best = test_loss < best_test_loss
        best_test_loss = min(test_loss, best_test_loss)
        save_checkpoint({
            'epoch': epoch,
            'best_test_loss': best_test_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)

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
