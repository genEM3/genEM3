import numpy as np
import ipywidgets as ipyw
import matplotlib.pyplot as plt

class ImageSliceViewer3D:
    """     
    ImageSliceViewer3D is for viewing volumetric image slices in jupyter or
    ipython notebooks. 
    
    User can interactively change the slice plane selection for the image and 
    the slice plane being viewed. 

    Argumentss:
    Volume = 3D input image
    figsize = default(8,8), to set the size of the figure
    cmap = default('plasma'), string for the matplotlib colormap. You can find 
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html
    
    Written by: Mohak Patel as a jupyter notebook here:
    https://github.com/mohakpatel/ImageSliceViewer3D
    
    """
    
    def __init__(self, volume, figsize=(8,8), cmap='gray'):
        self.volume = volume
        self.figsize = figsize
        self.cmap = cmap
        self.v = [np.min(volume), np.max(volume)]
        
        # Call to select slice plane
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
            options=['x-y','y-z', 'z-x'], value='x-y', 
            description='Slice plane selection:', disabled=False,
            style={'description_width': 'initial'}))
    
    def view_selection(self, view):
        # Transpose the volume to orient according to the slice plane selection
        orient = {"y-z":[1,2,0], "z-x":[2,0,1], "x-y": [0,1,2]}
        self.vol = np.transpose(self.volume, orient[view])
        maxZ = self.vol.shape[2] - 1
        
        # Call to view a slice within the selected slice plane
        ipyw.interact(self.plot_slice, 
            z=ipyw.IntSlider(min=0, max=maxZ, step=1, continuous_update=False, 
            description='Image Slice:'))
        
    def plot_slice(self, z):
        # Plot slice for the given plane and slice
        self.fig = plt.figure(figsize=self.figsize)
        plt.imshow(self.vol[:,:,z], cmap=plt.get_cmap(self.cmap), 
            vmin=self.v[0], vmax=self.v[1])

# Plotting input and output of individual training examples        
def data2fig_subplot(inputs, outputs, idx):
    fig, axs = plt.subplots(1, 2, figsize=(16,12))
    input_cpu = inputs[idx].data.cpu()
    img_input = input_cpu.numpy().squeeze()
    axs[0].imshow(img_input, cmap='gray')
    output_cpu = outputs[idx].data.cpu()
    img_output = output_cpu.numpy().squeeze()
    axs[1].imshow(img_output, cmap='gray')
    # AK: I need this line so that the figure is ported through X11 for me
    plt.show()
    return fig

# crop the valid part of the image
def crop_valid(tensor, center, width):
    valid_tensor = tensor[:, 0:1, center-width:center+width+1, center-width:center+width+1]
    return valid_tensor
