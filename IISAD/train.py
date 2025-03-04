"""
TRAIN GANOMALY

. Example: Run the following command from the terminal.
    run train.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIES
from __future__ import print_function

from options import Options
from lib.data import load_data
from lib.model import Ganomaly

##
def train():
    """ Training
    """

    ##
    # ARGUMENTS
    opt = Options().parse()
    ##
    print('进程数：', opt.workers)
    # LOAD DATA
    dataloader = load_data(opt)
    ##
    # LOAD MODEL
    mask_sizes = []
    mask_sizes.append([4])
    for mask_size in mask_sizes:
        model = Ganomaly(opt, dataloader, mask_size)
        # TRAIN MODEL
        model.train()
        del model

if __name__ == '__main__':
    train()
