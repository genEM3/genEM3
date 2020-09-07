"""Tensorboard related general functionalities"""
from tensorboard import program
import torch


def launch_tb(logdir: str = None, port: str = '7900'):
    """Launch the instance of tensorboard given the directory and port"""
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir, '--port', port])
    url = tb.launch()
    print(f'======\nLaunching tensorboard,\nDirectory: {logdir}\nPort: {port}\n======\n')
    return url


def add_graph(writer: torch.utils.tensorboard.SummaryWriter = None,
              model: torch.nn.Module = None,
              data_loader: torch.utils.data.dataloader = None,
              device: torch.device = torch.device('cpu')):
    """Plot the graph of the model"""
    # get an example image for running through the network
    input_batch_dict = next(iter(data_loader))
    input_batch = input_batch_dict['input'].to(device)
    writer.add_graph(model, input_batch)
