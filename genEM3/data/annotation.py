"""
Functions used in relation to data annotation that might not fit in other modules
"""
import os
import pickle
from typing import Sequence, Tuple
from collections import namedtuple
from functools import partial, partialmethod

import matplotlib.pyplot as plt
import numpy as np

from genEM3.data.wkwdata import DataSource, WkwData
from genEM3.util.path import get_data_dir

# Copied from pigeon data annotator:
import random
import functools
from IPython.display import display, clear_output
import ipywidgets as widgets


class Widget():
    """
    Build an interactive widget for annotating a list of input examples.

    Parameters
    ----------
    Returns
    -------
    """
    def __init__(self,
                 dataset: WkwData=None,
                 index_range: range=None,
                 button_names: Sequence[Sequence[str]] = [['No', 'Yes'], ['No', 'Yes']],
                 target_classes: Sequence[str] = ['Myelin', 'Debris'],
                 margin: int = 35, 
                 roi_size: int = 140):
        self.dataset = dataset
        self.index_range = index_range
        self._current_index = min(index_range)
        self.button_names = button_names
        self.target_classes = target_classes
        self.annotation_list = [(index, {cur_class: None for cur_class in target_classes}) for index in index_range]
        # margin and roi size for drawing each example
        self.margin = margin
        self.roi_size = roi_size
        # elements of the widget
        # index slider
        self.index_slider = self.get_slider()
        # output element for the example image (output of display_example)
        self.image_output = widgets.Output(layout={'border': '1px solid black', 'width': '50%', 
                                            'align_self':'center'})
        self.text_output = widgets.Label(value=None, layout=widgets.Layout(display="flex", justify_content="center", align_self='center',
                                                                width="35%", border="solid"))
        # Previous/Next buttons
        self.prev_next = self.get_prev_next_button()
        self.annotation_buttons = self.get_annotation_buttons()
        
        # Save/Load properties
        self.input_to_init = ['dataset','index_range','button_names','target_classes','margin','roi_size']
        self.additional_save_props = ['annotation_list', '_current_index']
    
    @property
    def current_index(self):
        return self._current_index
    
    @current_index.setter
    def current_index(self, current_index):
        # sync the slider with the current index
        assert min(self.index_range)<= current_index <= max(self.index_range), f'Index out of range. current: {current_index}, range{self.index_range}'
        self.index_slider.value = current_index
        self._current_index = current_index

    def display_current(self):
        """Display an image with a central rectangle for the roi"""
        _ , ax = plt.subplots(figsize=(10, 10))
        current_image = self.dataset.get_ordered_sample(self.current_index)['input'].squeeze()
        ax.imshow(current_image,cmap='gray')
        # Add a red rectangle
        rectangle = plt.Rectangle((self.margin, self.margin), 
                                   self.roi_size, self.roi_size, fill=False, ec="red")
        ax.add_patch(rectangle)
        plt.title(f'Sample index: {self.current_index}')
        # turn off the axis
        ax.axis('off')
        # output the plot
        plt.show()

    
    def update_image(self, relative_pos: int = 0):
        """
        Similar to the display_button_callback without the button input required by the ipywidgets
        """
        self.current_index += relative_pos
        self.update_result_text()
        with self.image_output:
            clear_output(wait=True)
            self.display_current()

    def update_result_text(self):
        """
        Update the text of the image to represent the current result for it
        """
        self.text_output.value = f'Annotation result: {self.annotation_list[self.current_index][1]}'

    def get_prev_next_button(self):
        b1 = widgets.Button(description="Previous")
        b2 = widgets.Button(description="Next")
        b1.on_click(partial(self.display_button_callback, widget_obj=self, relative_pos=-1))
        b2.on_click(partial(self.display_button_callback, widget_obj=self,relative_pos=1))
        control_buttons = widgets.HBox([widgets.Label('Controlers: '),b1, b2], layout=widgets.Layout(justify_content='center'))
        return control_buttons
    
    def get_slider(self):
        """
        Generate the slider for the sample indicees
        """
        progress = widgets.IntSlider(value=self.current_index,
                                     min=min(self.index_range),
                                     max=max(self.index_range),
                                     step=1,
                                     description='Index:',
                                     disabled=False,
                                     continuous_update=False,
                                     orientation='horizontal',
                                     layout=widgets.Layout(width='100%',justify_content='center'))
        # Change the current index and display the image when slider value is changed
        def on_change_handler(slider_change):
            # update current index, note: the current index setter also updates the value of the slider
            self.current_index = slider_change['new']
            self.update_image(relative_pos=0)
        progress.observe(on_change_handler, names='value')
        return progress
    
    def set_annotation(self, button_value, target_class):
        """
        Set the annotation value
        """
        self.annotation_list[self.current_index][1][target_class] = button_value

    def get_button(self, button_name, target_class):
        """Get the button given the button type [Yes or no] and target type [Debris, Myelin]"""
        assert button_name in ['Yes', 'No'], "Target type should be either 'Yes' or 'No'"
        button_value = 1.0 if (button_name == 'Yes') else 0.0
        button = widgets.Button(description=button_name,
                        disabled=False) 
        # Callback function of the button
        def on_click(b):
            self.set_annotation(button_value, target_class)
            self.update_result_text()
        button.on_click(on_click)
        return button

    def get_annotation_buttons(self):
        # Get the buttons for setting
        annotation_buttons = []
        for index, target in enumerate(self.target_classes):
            cur_target_buttons = [self.get_button(b_name, target) for b_name in self.button_names[index]]
            annotation_buttons.append(widgets.HBox([widgets.Label(target+': ')]+cur_target_buttons, layout=widgets.Layout(justify_content='center')))
        return annotation_buttons

    def show_widget(self):
        """
        Concatenate all the widgets and display them
        """
        # Load the current image
        self.update_image()
        all_widgets = widgets.VBox([self.index_slider] + self.annotation_buttons + [self.prev_next, self.text_output, self.image_output])
        display(all_widgets)

    def save(self, file_name):
        """
        Save the dictionary of the object to the filename
        Note: The ipywidget items seem to be not picklable so I just save the necessary items for initiating for load
        """
        
        things2save = self.input_to_init + self.additional_save_props 
        with open(file_name, 'wb') as output:
            dict2save = {k: v for k, v in self.__dict__.items() if k in things2save}
            pickle.dump(dict2save, output, pickle.HIGHEST_PROTOCOL)

    def update_annotation_from_multiclass(self):
        """
        Use the multiclass annotation in the dataset to update the annotation_list
        """
        multiclass_targets = [self.dataset.get_target_from_sample_idx(i) for i in self.index_range]
        for index, item in enumerate(multiclass_targets):
            item = int(item)
            if item == 0:
                self.annotation_list[index][1]['Myelin'] = 0.0
                self.annotation_list[index][1]['Debris'] = 0.0
            elif item == 1:
                self.annotation_list[index][1]['Myelin'] = 0.0
                self.annotation_list[index][1]['Debris'] = 1.0
            elif item == 2:
                self.annotation_list[index][1]['Myelin'] = 1.0
                self.annotation_list[index][1]['Debris'] = 0.0
            else:
                raise ValueError

    def update_from_json_dataset(self):
        """
        Update the targets from the dataset used to define the widget
        """
        targets = [self.dataset.get_target_from_sample_idx(i) for i in self.index_range]
        for index, item in enumerate(targets):
            self.annotation_list[index][1]['Debris'] = item[0]
            self.annotation_list[index][1]['Myelin'] = item[1]
            
    @classmethod
    def load(cls, file_name):
        """Load the dictionary of input and the initialize the object from it"""
        propertiesAfterLoading = ['annotation_list', '_current_index']
        with open(file_name, 'rb') as input:
            saved_dict = pickle.load(input)
            kwargs = {k: v for k, v in saved_dict.items() if k not in propertiesAfterLoading}
            loaded_widget = cls(**kwargs)
        # set the annotation list and the current index as well
        loaded_widget.annotation_list = saved_dict.get('annotation_list')
        loaded_widget.current_index = saved_dict.get('_current_index')
        return loaded_widget


    @staticmethod 
    def display_button_callback(btn, widget_obj=None, relative_pos=0):
        """
        Update the output image given the relative change as compared to current index
        Note: The reason for the use of static method is the need for the intial argument to be the btn representation from ipywidgets
        See here: https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Events.html
        """
        widget_obj.current_index += relative_pos
        widget_obj.update_result_text()
        with widget_obj.image_output:
            clear_output(wait=True)# wait until the next image is loaded
            widget_obj.display_current()
        
def update_data_source_targets(dataset: WkwData,
                               target_index_tuple_list: Sequence[Tuple[int, float]]):
    """Create an updated list of datasources from a wkwdataset and a list of sample index, target_class pair"""
    list_source_idx = [dataset.get_source_idx_from_sample_idx(sample_idx) for (sample_idx, _) in target_index_tuple_list]
    source_list = []
    for cur_target_tuple in target_index_tuple_list:
        cur_target = cur_target_tuple[1]
        sample_idx = cur_target_tuple[0]
        s_index = list_source_idx[sample_idx]
        s = dataset.data_sources[s_index]
        source_list.append(DataSource(id=s.id, input_path=s.input_path, input_bbox=s.input_bbox,
                                      input_mean=s.input_mean, input_std=s.input_std, target_path=s.target_path,
                                      target_bbox=s.target_bbox, target_class=cur_target, target_binary=s.target_binary))
    return source_list


def update_data_source_bbox(dataset: WkwData,
                            bbox_list: Sequence[Tuple[int, Sequence[int]]]):
    """Create an updated list of datasources from a wkwdataset and a list of index and bounding box tuples"""
    assert len(bbox_list) == len(dataset)
    source_list = []
    for sample_idx, (source_idx, cur_bbox) in enumerate(bbox_list):
        s = dataset.data_sources[source_idx]
        source_list.append(DataSource(id=str(sample_idx), input_path=s.input_path, input_bbox=cur_bbox,
                                      input_mean=s.input_mean, input_std=s.input_std, target_path=s.target_path,
                                      target_bbox=cur_bbox, target_class=s.target_class, target_binary=s.target_binary))
    return source_list


def display_example(index: int, dataset: WkwData, margin: int = 35, roi_size: int = 140):
    """Display an image with a central rectangle for the roi"""
    _ , ax = plt.subplots(figsize=(10, 10))
    ax.imshow(dataset.get_ordered_sample(index)['input'].squeeze(),cmap='gray')
    rectangle = plt.Rectangle((margin,margin), roi_size, roi_size, fill=False, ec="red")
    ax.add_patch(rectangle)
    ax.axis('off')
    plt.show()


def merge_json_from_data_dir(fnames: Sequence[str], output_fname: str):
    """Function concatenates the data directory to the list of file names and concatenats the related jsons"""
    # Test concatenating jsons
    full_fnames = []
    for fname in fnames:
        full_fname = os.path.join(get_data_dir(), fname)
        full_fnames.append(full_fname)

    # Concatenate the test and training data sets
    full_output_name = os.path.join(get_data_dir(), output_fname)
    all_ds = WkwData.concat_datasources(json_paths_in=full_fnames, json_path_out=full_output_name)
    return all_ds


def patch_source_list_from_dataset(dataset: WkwData, 
                                   margin: int = 35,
                                   roi_size: int = 140):
    """Return two data_sources from the image patches contained in a dataset. One data source has a larger bbox for annotations"""
    corner_xy_index = [0, 1]
    length_xy_index = [3, 4]
    large_bboxes_idx = []
    bboxes_idx = []
    for idx in range(len(dataset)):
        (source_idx, original_cur_bbox) = dataset.get_bbox_for_sample_idx(idx)
        bboxes_idx.append((source_idx, original_cur_bbox))
        cur_bbox = np.asarray(original_cur_bbox)
        cur_bbox[corner_xy_index] = cur_bbox[corner_xy_index] - margin
        cur_bbox[length_xy_index] = cur_bbox[length_xy_index] + margin*2
        # large bbox append
        large_bboxes_idx.append((source_idx, cur_bbox.tolist()))

    assert len(large_bboxes_idx) == len(dataset) == len(bboxes_idx)
    large_source_list = update_data_source_bbox(dataset, large_bboxes_idx)
    patch_source_list = update_data_source_bbox(dataset, bboxes_idx)
    return {'original': patch_source_list,'large':large_source_list}


def divide_range(total_size: int, chunk_size: int = 1000,):
    """Break down the range into partitions of 1000"""
    chunk_size = 1000
    num_thousand, remainder = divmod(total_size, chunk_size)
    list_ranges = []
    # Create a list of ranges
    for i in range(num_thousand):
        list_ranges.append(range(i*chunk_size, (i+1)*chunk_size))
    if remainder > 0:
        final_range = range(num_thousand*chunk_size, num_thousand*chunk_size+remainder)
        list_ranges.append(final_range)

    return list_ranges