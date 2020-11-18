"""
Functions used in relation to data annotation that might not fit in other modules
"""
import os
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from genEM3.data.wkwdata import DataSource, WkwData
from genEM3.util.path import get_data_dir

# Copied from pigeon data annotator:
import random
import functools
from IPython.display import display, clear_output
from ipywidgets import Button, Dropdown, HTML, HBox, IntSlider, FloatSlider, Textarea, Output

def annotate(examples,
             options=None,
             shuffle=False,
             include_skip=True,
             display_fn=display):
    """
    Build an interactive widget for annotating a list of input examples.

    Parameters
    ----------
    examples: list(any), list of items to annotate
    options: list(any) or tuple(start, end, [step]) or None
             if list: list of labels for binary classification task (Dropdown or Buttons)
             if tuple: range for regression task (IntSlider or FloatSlider)
             if None: arbitrary text input (TextArea)
    shuffle: bool, shuffle the examples before annotating
    include_skip: bool, include option to skip example while annotating
    display_fn: func, function for displaying an example to the user

    Returns
    -------
    annotations : list of tuples, list of annotated examples (example, label)
    """
    examples = list(examples)
    if shuffle:
        random.shuffle(examples)

    annotations = []
    current_index = -1

    def set_label_text():
        nonlocal count_label
        count_label.value = '{} examples annotated, {} examples left'.format(
            len(annotations), len(examples) - current_index
        )

    def show_next():
        nonlocal current_index
        current_index += 1
        set_label_text()
        if current_index >= len(examples):
            for btn in buttons:
                btn.disabled = True
            print('Annotation done.')
            return
        with out:
            clear_output(wait=True)
            display_fn(examples[current_index])

    def add_annotation(annotation):
        annotations.append((examples[current_index], annotation))
        show_next()

    def skip(btn):
        show_next()

    count_label = HTML()
    set_label_text()
    display(count_label)

    if type(options) == list:
        task_type = 'classification'
    elif type(options) == tuple and len(options) in [2, 3]:
        task_type = 'regression'
    elif options is None:
        task_type = 'captioning'
    else:
        raise Exception('Invalid options')

    buttons = []
    
    if task_type == 'classification':
        use_dropdown = len(options) > 5

        if use_dropdown:
            dd = Dropdown(options=options)
            display(dd)
            btn = Button(description='submit')
            def on_click(btn):
                add_annotation(dd.value)
            btn.on_click(on_click)
            buttons.append(btn)
        
        else:
            for label in options:
                btn = Button(description=label)
                def on_click(label, btn):
                    add_annotation(label)
                btn.on_click(functools.partial(on_click, label))
                buttons.append(btn)

    elif task_type == 'regression':
        target_type = type(options[0])
        if target_type == int:
            cls = IntSlider
        else:
            cls = FloatSlider
        if len(options) == 2:
            min_val, max_val = options
            slider = cls(min=min_val, max=max_val)
        else:
            min_val, max_val, step_val = options
            slider = cls(min=min_val, max=max_val, step=step_val)
        display(slider)
        btn = Button(description='submit')
        def on_click(btn):
            add_annotation(slider.value)
        btn.on_click(on_click)
        buttons.append(btn)

    else:
        ta = Textarea()
        display(ta)
        btn = Button(description='submit')
        def on_click(btn):
            add_annotation(ta.value)
        btn.on_click(on_click)
        buttons.append(btn)

    if include_skip:
        btn = Button(description='skip')
        btn.on_click(skip)
        buttons.append(btn)

    box = HBox(buttons)
    display(box)

    out = Output()
    display(out)

    show_next()

    return annotations

def update_data_source_targets(dataset: WkwData,
                               target_index_tuple_list: Sequence[Tuple[int, float]]):
    """Create an updated list of datasources from a wkwdataset and a list of sample index, target_class pair"""
    list_source_idx = [dataset.get_source_idx_from_sample_idx(sample_idx) for (sample_idx, _) in target_index_tuple_list]
    source_list = []
    for i, cur_target_tuple in enumerate(target_index_tuple_list):
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
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(dataset.get_ordered_sample(index)['input'].squeeze(),cmap='gray')
    rectangle = plt.Rectangle((margin,margin), roi_size, roi_size, fill=False, ec="red")
    ax.add_patch(rectangle)
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
    larger_sources = update_data_source_bbox(dataset, large_bboxes_idx)
    patch_source_list = update_data_source_bbox(dataset, bboxes_idx)
    return {'original': patch_source_list,'large':larger_sources}


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
