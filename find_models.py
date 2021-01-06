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
from ofa.tutorial import evaluate_ofa_subnet, evaluate_ofa_specialized, evaluate_ofa_space, evaluate_ofa_best_acc_team, evaluate_ofa_random_sample

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
    net_id = evaluate_ofa_random_sample(imagenet_data_path, data_loader, ensemble=True)
    # print('Finished evaluating the pretrained sub-network: %s!' % net_id)
    print("best ensemble team{}".format(net_id))
else:
    print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')

accuracy_predictor = AccuracyPredictor(
    pretrained=True,
    device='cuda:0' if cuda_available else 'cpu'
)

print('The accuracy predictor is ready!')
print(accuracy_predictor.model)

nets = []
for i in range(100):
    ofa_network.sample_active_subnet()
    subnet = ofa_network.get_active_subnet(preserve_weight=True)
    net_config = subnet.config
    top1 = evaluate_ofa_subnet(
        ofa_network,
        imagenet_data_path,
        net_config,
        data_loader,
        batch_size=250,
        device='cuda:0' if cuda_available else 'cpu')
    print("net_config:{} top1:{}".format(net_config, top1))
    if top1>=77 and top1<=79:
        nets.append(net_config)

fh = open(os.path.join(args.output_path, 'ofa_nets.json'), 'w')
json.dump(nets, fh)
fh.close()

# target_hardware = 'v100'
# latency_table = LatencyTable(device=target_hardware)
# print('The Latency lookup table on %s is ready!' % target_hardware)
#
# latency_constraint = 25  # ms, suggested range [15, 33] ms
# P = 100  # The size of population in each generation
# N = 500  # How many generations of population to be searched
# r = 0.25  # The ratio of networks that are used as parents for next generation
# params = {
#     'constraint_type': target_hardware, # Let's do FLOPs-constrained search
#     'efficiency_constraint': latency_constraint,
#     'mutate_prob': 0.1, # The probability of mutation in evolutionary search
#     'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
#     'efficiency_predictor': latency_table, # To use a predefined efficiency predictor.
#     'accuracy_predictor': accuracy_predictor, # To use a predefined accuracy_predictor predictor.
#     'population_size': P,
#     'max_time_budget': N,
#     'parent_ratio': r,
# }
#
# # build the evolution finder
# finder = EvolutionFinder(**params)
#
# # start searching
# result_lis = []
# st = time.time()
# best_valids, best_info = finder.run_evolution_search()
# result_lis.append(best_info)
# ed = time.time()
# print('Found best architecture on %s with latency <= %.2f ms in %.2f seconds! '
#       'It achieves %.2f%s predicted accuracy with %.2f ms latency on %s.' %
#       (target_hardware, latency_constraint, ed-st, best_info[0] * 100, '%', best_info[-1], target_hardware))
#
# # visualize the architecture of the searched sub-net
# _, net_config, latency = best_info
# ofa_network.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
# print('Architecture of the searched sub-net:')
# print(ofa_network.module_str)
