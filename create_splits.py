import argparse
import glob
import os
import random
import shutil
import numpy as np

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    
    data_fnames = glob.glob(data_dir + os.sep + 'training_and_validation' + os.sep + '*.tfrecord')
    
    # ======
    train_data_dir = data_dir + os.sep + 'train'
    val_data_dir = data_dir + os.sep + 'val'
    
    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(val_data_dir, exist_ok=True)
    
    # ======== split data ==============
    random.shuffle(data_fnames)
    train_proportion = int(len(data_fnames) * .8)
    train_fnames = data_fnames[0: train_proportion]
    val_fnames = list(set(data_fnames).difference(train_fnames))
    
    # ============ move data =========
    
    # move train data
    for fname in train_fnames:
        new_fname = fname.split(os.sep)[-1]
        shutil.move(fname, train_data_dir + os.sep + new_fname)
        
    # move val data
    for fname in val_fnames:
        new_fname = fname.split(os.sep)[-1]
        shutil.move(fname, val_data_dir + os.sep + new_fname)
                     
                     

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)