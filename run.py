import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
import time
import random
import math
import copy
from matplotlib import pyplot as plt

from ofa.model_zoo import ofa_net
from ofa.utils import download_url

# from ofa.tutorial.accuracy_predictor import AccuracyPredictor
# from ofa.tutorial.flops_table import FLOPsTable
# from ofa.tutorial.latency_table import LatencyTable
# from ofa.tutorial.evolution_finder import EvolutionFinder
# from ofa.tutorial.imagenet_eval_helper import evaluate_ofa_subnet, evaluate_ofa_specialized
from ofa.tutorial import AccuracyPredictor, FLOPsTable, LatencyTable, EvolutionFinder
from ofa.tutorial import evaluate_ofa_subnet, evaluate_ofa_specialized, evaluate_ofa_space

# set random seed
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!'%random_seed)
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(random_seed)
    print('Using GPU.')
else:
    print('Using CPU.')
ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=True)
print('The OFA Network is ready.')
if cuda_available:
    # path to the ImageNet dataset
    print("Please input the path to the ImageNet dataset.\n")
    imagenet_data_path = "/gdata/ImageNet2012"

    print(os.path.isdir(imagenet_data_path))
    # if 'imagenet_data_path' is empty, download a subset of ImageNet containing 2000 images (~250M) for test
    # if not os.path.isdir(imagenet_data_path):
    #     os.makedirs(imagenet_data_path, exist_ok=True)
    #     download_url('https://hanlab.mit.edu/files/OnceForAll/ofa_cvpr_tutorial/imagenet_1k.zip', model_dir='data')
    #     ! cd data && unzip imagenet_1k 1>/dev/null && cd ..
    #     ! cp -r data/imagenet_1k/* $imagenet_data_path
    #     ! rm -rf data
    #     print('%s is empty. Download a subset of ImageNet for test.' % imagenet_data_path)

    print('The ImageNet dataset files are ready.')
else:
    print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')
if cuda_available:
    # The following function build the data transforms for test
    def build_val_transform(size):
        return transforms.Compose([
            transforms.Resize(int(math.ceil(size / 0.875))),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root=os.path.join(imagenet_data_path, 'val'),
            transform=build_val_transform(224)
        ),
        batch_size=250,  # test batch size
        shuffle=True,
        num_workers=16,  # number of workers for the data loader
        pin_memory=True,
        drop_last=False,
    )
    print('The ImageNet dataloader is ready.')
else:
    data_loader = None
    print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')
if cuda_available:
    net_id = evaluate_ofa_space(imagenet_data_path, data_loader, ensemble=True)
    # print('Finished evaluating the pretrained sub-network: %s!' % net_id)
    print("best ensemble team{}".format(net_id))
else:
    print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')