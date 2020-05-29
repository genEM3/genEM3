import os
from torch.utils.tensorboard import SummaryWriter

class Trainer:

    def __init__(self,
                 run_root,
                 dataloader,
                 model,
                 optimizer,
                 criterion,
                 num_epoch,
                 log_int=100,
                 device='cpu'
                 ):

        self.run_root = run_root
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epoch = num_epoch
        self.log_int = log_int
        self.device = device

        self.log_root = os.path.join(run_root, '.log')

    def train(self):

        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        writer = SummaryWriter(self.log_root)
        writer_int = 5
        n_epoch = 500
        it = 0

        for epoch in range(self.num_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.dataloader):
                it += 1

                if which_device == 'gpu':
                    inputs = data.to(device)
                    labels = data.clone().to(device)
                else:
                    inputs = data
                    labels = data.clone()

                #         data = self.dataloader.dataset[0]
                #         inputs = data.unsqueeze(0)
                #         labels = data.clone().unsqueeze(0)

                #         labels_valid = crop_valid(labels, input_center, valid_width)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                #         outputs_valid = crop_valid(outputs, input_center, valid_width)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if (i + 1) % writer_int == 0:
                    print('it: {} (epoch: {}, batch: {}), running loss: {:0.3f}'.format(i, epoch, i + 1, running_loss))

                    writer.add_scalar('loss', loss.item(), it)
                    writer.add_scalar('running_loss', running_loss, it)
                    writer.add_figure('inputs', data2fig_subplot(inputs, outputs, 0), it)

                    running_loss = 0.0