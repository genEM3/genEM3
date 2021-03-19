"""
Functions for reading and writing to disk
"""
import pickle


def save_dict(dictionary: dict, fname: str):
    """
    Save the dictionary to the memory

    Args:
        dictionary: the input dictionary
        fname: file name to save
    Returns:
        None
    """
    with open(fname, 'wb') as fp:
        pickle.dump(dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict(fname: str):
    """
    Load the dictionary from the pickled file
    
    Args: 
        fname: full file name
    Returns:
        dictionary: loaded dictionary
    """
    with open(fname, 'rb') as fp:
        dictionary = pickle.load(fp)
    return dictionary