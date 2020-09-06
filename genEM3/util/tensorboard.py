"""Tensorboard related general functionalities"""
from tensorboard import program


def launch_tb(logdir: str = None, port: str = '7900'):
    """Launch the instance of tensorboard given the directory and port"""
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir, '--port', port])
    url = tb.launch()
    print(f'======\nLaunching tensorboard,\nDirectory: {logdir}\nPort: {port}\n======\n')
    return url
