import os.path as osp
import numpy as np
import math
from tqdm import tqdm

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms, datasets

from ofa.utils import AverageMeter, accuracy
from ofa.model_zoo import ofa_specialized
from ofa.imagenet_classification.elastic_nn.utils import set_running_statistics
import copy
import random


def evaluate_ofa_subnet(ofa_net, path, net_config, data_loader, batch_size, device='cuda:0'):
    assert 'ks' in net_config and 'd' in net_config and 'e' in net_config
    assert len(net_config['ks']) == 20 and len(net_config['e']) == 20 and len(net_config['d']) == 5
    ofa_net.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
    subnet = ofa_net.get_active_subnet().to(device)
    calib_bn(subnet, path, net_config['r'][0], batch_size)
    top1 = validate(subnet, path, net_config['r'][0], data_loader, batch_size, device)
    return top1


def calib_bn(net, path, image_size, batch_size, num_images=2000):
    # print('Creating dataloader for resetting BN running statistics...')
    dataset = datasets.ImageFolder(
        osp.join(
            path,
            'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406],
                std=[
                    0.229,
                    0.224,
                    0.225]
            ),
        ])
    )
    chosen_indexes = np.random.choice(list(range(len(dataset))), num_images)
    sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sub_sampler,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )
    # print('Resetting BN running statistics (this may take 10-20 seconds)...')
    set_running_statistics(net, data_loader)


def ensemble_validate(nets, path, image_size, data_loader, batch_size=100, device='cuda:0'):
    if 'cuda' in device:
        for net in nets:
            net = torch.nn.DataParallel(net).to(device)
    else:
        for net in nets:
            net = net.to(device)

    data_loader.dataset.transform = transforms.Compose([
        transforms.Resize(int(math.ceil(image_size / 0.875))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().to(device)

    for net in nets:
        net.eval()
        net = net.to(device)
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        with tqdm(total=len(data_loader), desc='Validate') as t:
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(device), labels.to(device)
                # compute output
                n = len(nets)
                output = 0
                for i, net in enumerate(nets):
                    if i == 0:
                        output =net(images)
                    else:
                        output+=net(images)
                output = output/n
                loss = criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                t.set_postfix({
                    'loss': losses.avg,
                    'top1': top1.avg,
                    'top5': top5.avg,
                    'img_size': images.size(2),
                })
                t.update(1)

    print('Results: loss=%.5f,\t top1=%.3f,\t top5=%.1f' % (losses.avg, top1.avg, top5.avg))
    return top1.avg


def validate(net, path, image_size, data_loader, batch_size=100, device='cuda:0'):
    if 'cuda' in device:
        net = torch.nn.DataParallel(net).to(device)
    else:
        net = net.to(device)

    data_loader.dataset.transform = transforms.Compose([
        transforms.Resize(int(math.ceil(image_size / 0.875))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().to(device)

    net.eval()
    net = net.to(device)
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        with tqdm(total=len(data_loader), desc='Validate') as t:
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(device), labels.to(device)
                # compute output
                output = net(images)
                loss = criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                t.set_postfix({
                    'loss': losses.avg,
                    'top1': top1.avg,
                    'top5': top5.avg,
                    'img_size': images.size(2),
                })
                t.update(1)

    
    print('Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f' % (losses.avg, top1.avg, top5.avg))
    return top1.avg
    

def evaluate_ofa_specialized(path, data_loader, batch_size=100, device='cuda:0', ensemble=False):
    def select_platform_name():
        valid_platform_name = [
            'pixel1', 'pixel2', 'note10', 'note8', 's7edge', 'lg-g8', '1080ti', 'v100', 'tx2', 'cpu', 'flops'
        ]
        
        print("Please select a hardware platform from ('pixel1', 'pixel2', 'note10', 'note8', 's7edge', 'lg-g8', '1080ti', 'v100', 'tx2', 'cpu', 'flops')!\n")
        
        while True:
            platform_name = input()
            platform_name = platform_name.lower()
            if platform_name in valid_platform_name:
                return platform_name
            print("Platform name is invalid! Please select in ('pixel1', 'pixel2', 'note10', 'note8', 's7edge', 'lg-g8', '1080ti', 'v100', 'tx2', 'cpu', 'flops')!\n")
    
    def select_netid(platform_name):
        platform_efficiency_map = {
            'pixel1': {
                143: 'pixel1_lat@143ms_top1@80.1_finetune@75',
                132: 'pixel1_lat@132ms_top1@79.8_finetune@75',
                79: 'pixel1_lat@79ms_top1@78.7_finetune@75',
                58: 'pixel1_lat@58ms_top1@76.9_finetune@75',
                40: 'pixel1_lat@40ms_top1@74.9_finetune@25',
                28: 'pixel1_lat@28ms_top1@73.3_finetune@25',
                20: 'pixel1_lat@20ms_top1@71.4_finetune@25',
            },
            
            'pixel2': {
                62: 'pixel2_lat@62ms_top1@75.8_finetune@25',
                50: 'pixel2_lat@50ms_top1@74.7_finetune@25',
                35: 'pixel2_lat@35ms_top1@73.4_finetune@25',
                25: 'pixel2_lat@25ms_top1@71.5_finetune@25',
            },
            
            'note10': {
                64: 'note10_lat@64ms_top1@80.2_finetune@75',
                50: 'note10_lat@50ms_top1@79.7_finetune@75',
                41: 'note10_lat@41ms_top1@79.3_finetune@75',
                30: 'note10_lat@30ms_top1@78.4_finetune@75',
                22: 'note10_lat@22ms_top1@76.6_finetune@25',
                16: 'note10_lat@16ms_top1@75.5_finetune@25',
                11: 'note10_lat@11ms_top1@73.6_finetune@25',
                8: 'note10_lat@8ms_top1@71.4_finetune@25',
            },
            
            'note8': {
                65: 'note8_lat@65ms_top1@76.1_finetune@25',
                49: 'note8_lat@49ms_top1@74.9_finetune@25',
                31: 'note8_lat@31ms_top1@72.8_finetune@25',
                22: 'note8_lat@22ms_top1@70.4_finetune@25',
            },
            
            's7edge': {
                88: 's7edge_lat@88ms_top1@76.3_finetune@25',
                58: 's7edge_lat@58ms_top1@74.7_finetune@25',
                41: 's7edge_lat@41ms_top1@73.1_finetune@25',
                29: 's7edge_lat@29ms_top1@70.5_finetune@25',
            },
            
            'lg-g8': {
                24: 'LG-G8_lat@24ms_top1@76.4_finetune@25',
                16: 'LG-G8_lat@16ms_top1@74.7_finetune@25',
                11: 'LG-G8_lat@11ms_top1@73.0_finetune@25',
                8: 'LG-G8_lat@8ms_top1@71.1_finetune@25',
            },
            
            '1080ti': {
                27: '1080ti_gpu64@27ms_top1@76.4_finetune@25',
                22: '1080ti_gpu64@22ms_top1@75.3_finetune@25',
                15: '1080ti_gpu64@15ms_top1@73.8_finetune@25',
                12: '1080ti_gpu64@12ms_top1@72.6_finetune@25',
            },
            
            'v100': {
                11: 'v100_gpu64@11ms_top1@76.1_finetune@25',
                9: 'v100_gpu64@9ms_top1@75.3_finetune@25',
                6: 'v100_gpu64@6ms_top1@73.0_finetune@25',
                5: 'v100_gpu64@5ms_top1@71.6_finetune@25',
            },
            
            'tx2': {
                96: 'tx2_gpu16@96ms_top1@75.8_finetune@25',
                80: 'tx2_gpu16@80ms_top1@75.4_finetune@25',
                47: 'tx2_gpu16@47ms_top1@72.9_finetune@25',
                35: 'tx2_gpu16@35ms_top1@70.3_finetune@25',
            },
            
            'cpu': {
                17: 'cpu_lat@17ms_top1@75.7_finetune@25',
                15: 'cpu_lat@15ms_top1@74.6_finetune@25',
                11: 'cpu_lat@11ms_top1@72.0_finetune@25',
                10: 'cpu_lat@10ms_top1@71.1_finetune@25',
            },
            
            'flops': {
                595: 'flops@595M_top1@80.0_finetune@75',
                482: 'flops@482M_top1@79.6_finetune@75',
                389: 'flops@389M_top1@79.1_finetune@75',
            }
        }
        
        sub_efficiency_map = platform_efficiency_map[platform_name]
        if not platform_name == 'flops':
            print("Now, please specify a latency constraint for model specialization among", sorted(list(sub_efficiency_map.keys())), 'ms. (Please just input the number.) \n')
        else:
            print("Now, please specify a FLOPs constraint for model specialization among", sorted(list(sub_efficiency_map.keys())), 'MFLOPs. (Please just input the number.) \n')
        
        while True:
            efficiency_constraint = input()
            if not efficiency_constraint.isdigit():
                print('Sorry, please input an integer! \n')
                continue
            efficiency_constraint = int(efficiency_constraint)
            if not efficiency_constraint in sub_efficiency_map.keys():
                print('Sorry, please choose a value from: ', sorted(list(sub_efficiency_map.keys())), '.\n')
                continue
            return sub_efficiency_map[efficiency_constraint]



    if not ensemble:
        platform_name = select_platform_name()
        net_id = select_netid(platform_name)
        net, image_size = ofa_specialized(net_id=net_id, pretrained=True)
        validate(net, path, image_size, data_loader, batch_size, device)
    else:
        nets = []
        for i in range(2):
            print('{}model'.format(i))
            platform_name = select_platform_name()
            net_id = select_netid(platform_name)
            net, image_size = ofa_specialized(net_id=net_id, pretrained=True)
            nets.append(net)
        ensemble_validate(nets, path, image_size, data_loader, batch_size, device)


    return net_id


net_id = ['pixel1_lat@143ms_top1@80.1_finetune@75', 'pixel1_lat@132ms_top1@79.8_finetune@75',
          'pixel1_lat@79ms_top1@78.7_finetune@75', 'pixel1_lat@58ms_top1@76.9_finetune@75',
          'pixel1_lat@40ms_top1@74.9_finetune@25', 'pixel1_lat@28ms_top1@73.3_finetune@25',
          'pixel1_lat@20ms_top1@71.4_finetune@25', 'pixel2_lat@62ms_top1@75.8_finetune@25',
          'pixel2_lat@50ms_top1@74.7_finetune@25', 'pixel2_lat@35ms_top1@73.4_finetune@25',
          'pixel2_lat@25ms_top1@71.5_finetune@25', 'note10_lat@64ms_top1@80.2_finetune@75',
          'note10_lat@50ms_top1@79.7_finetune@75', 'note10_lat@41ms_top1@79.3_finetune@75',
          'note10_lat@16ms_top1@75.5_finetune@25', 'note10_lat@11ms_top1@73.6_finetune@25',
          'note10_lat@8ms_top1@71.4_finetune@25', 'note8_lat@65ms_top1@76.1_finetune@25',
          'note8_lat@49ms_top1@74.9_finetune@25', 'note8_lat@31ms_top1@72.8_finetune@25',
          'note8_lat@22ms_top1@70.4_finetune@25', 's7edge_lat@88ms_top1@76.3_finetune@25',
          's7edge_lat@58ms_top1@74.7_finetune@25', 's7edge_lat@41ms_top1@73.1_finetune@25',
          's7edge_lat@29ms_top1@70.5_finetune@25', 'LG-G8_lat@24ms_top1@76.4_finetune@25',
          'LG-G8_lat@16ms_top1@74.7_finetune@25', 'LG-G8_lat@11ms_top1@73.0_finetune@25',
          'LG-G8_lat@8ms_top1@71.1_finetune@25', '1080ti_gpu64@27ms_top1@76.4_finetune@25',
          '1080ti_gpu64@22ms_top1@75.3_finetune@25', '1080ti_gpu64@15ms_top1@73.8_finetune@25',
          '1080ti_gpu64@12ms_top1@72.6_finetune@25', 'v100_gpu64@11ms_top1@76.1_finetune@25',
          'v100_gpu64@9ms_top1@75.3_finetune@25', 'v100_gpu64@6ms_top1@73.0_finetune@25',
          'v100_gpu64@5ms_top1@71.6_finetune@25', 'tx2_gpu16@96ms_top1@75.8_finetune@25',
          'tx2_gpu16@80ms_top1@75.4_finetune@25', 'tx2_gpu16@47ms_top1@72.9_finetune@25',
          'tx2_gpu16@35ms_top1@70.3_finetune@25', 'cpu_lat@17ms_top1@75.7_finetune@25',
          'cpu_lat@15ms_top1@74.6_finetune@25', 'cpu_lat@11ms_top1@72.0_finetune@25',
          'cpu_lat@10ms_top1@71.1_finetune@25', 'flops@595M_top1@80.0_finetune@75',
          'flops@482M_top1@79.6_finetune@75', 'flops@389M_top1@79.1_finetune@75', ]

def evaluate_ofa_space(path, data_loader, batch_size=100, device='cuda:0', ensemble=False):
    net_acc=[]
    for i, id in enumerate(net_id):
        acc=""
        for j in range(2, len(id)):
            if id[j]=='.':
                acc=id[j-2]+id[j-1]+id[j]+id[j+1]
        net_acc.append(acc)
    id =np.argsort(np.array(net_acc))
    new_net_id = copy.deepcopy(net_id)
    for i, sortid in enumerate(id):
        new_net_id[i] = net_id[sortid]
    print('new_net_id', new_net_id)
    n = len(net_id)
    best_acc = 0
    space = []
    best_team =[]
    for i in range(1, n):
        for j in range(i):
            nets = []
            team = []
            team.append(j)
            team.append(i)
            net, image_size = ofa_specialized(net_id=new_net_id[j], pretrained=True)
            nets.append(net)
            net, image_size = ofa_specialized(net_id=new_net_id[i], pretrained=True)
            nets.append(net)
            acc = ensemble_validate(nets, path, image_size, data_loader, batch_size, device)
            if acc>best_acc:
                best_acc=acc
                best_team = team
        print('space {} best_acc{}'.format(i+1, best_acc))
        space.append(best_acc)
    print('space:{}'.format(space))
    return net_id[best_team[0]], net_id[best_team[1]]


def evaluate_ofa_best_acc_team(path, data_loader, batch_size=100, device='cuda:0', ensemble=False):
    net_acc=[]
    for i, id in enumerate(net_id):
        acc=""
        for j in range(2, len(id)):
            if id[j]=='.':
                acc=id[j-2]+id[j-1]+id[j]+id[j+1]
        net_acc.append(acc)
    id =np.argsort(np.array(net_acc))
    new_net_id = copy.deepcopy(net_id)
    for i, sortid in enumerate(id):
        new_net_id[i] = net_id[sortid]
    print('new_net_id', new_net_id)
    n = len(net_id)
    best_acc = 0
    space = []
    best_team =[]
    i = n-1
    for j in range(18, n):
        nets = []
        team = []
        team.append(j)
        team.append(i)
        net, image_size = ofa_specialized(net_id=new_net_id[j], pretrained=True)
        nets.append(net)
        net, image_size = ofa_specialized(net_id=new_net_id[i], pretrained=True)
        nets.append(net)
        acc = ensemble_validate(nets, path, image_size, data_loader, batch_size, device)
        print('net i:{} netj:{} acc:{}'.format(new_net_id[i], new_net_id[j], acc))
        if acc>best_acc:
            best_acc=acc
            best_team = team

    print('space {} best_acc{}'.format(i+1, best_acc))
    space.append(best_acc)
    print('space:{}'.format(space))
    return new_net_id[best_team[0]], new_net_id[best_team[1]]


def evaluate_ofa_random_sample(path, data_loader, batch_size=100, device='cuda:0', ensemble=False):
    net_acc=[]
    for i, id in enumerate(net_id):
        acc=""
        for j in range(2, len(id)):
            if id[j]=='.':
                acc=id[j-2]+id[j-1]+id[j]+id[j+1]
        net_acc.append(acc)
    id =np.argsort(np.array(net_acc))
    new_net_id = copy.deepcopy(net_id)
    for i, sortid in enumerate(id):
        new_net_id[i] = net_id[sortid]
    print('new_net_id', new_net_id)
    n = len(net_id)
    best_acc = 0
    acc_list = []
    space = []
    best_team =[]
    for k in range(20):
        nets = []
        team = []
        i = random.randint(0, n-1)
        j = (i + random.randint(1, n-1)) % n
        print('i:{} j:{}'.format(i, j))
        team.append(j)
        team.append(i)
        net, image_size = ofa_specialized(net_id=new_net_id[j], pretrained=True)
        nets.append(net)
        net, image_size = ofa_specialized(net_id=new_net_id[i], pretrained=True)
        nets.append(net)
        acc = ensemble_validate(nets, path, image_size, data_loader, batch_size, device)
        print('net i:{} netj:{} acc:{}'.format(new_net_id[i], new_net_id[j], acc))
        acc_list.append(acc)
        if acc>best_acc:
            best_acc=acc
            best_team = team
    avg_acc = np.mean(acc_list)
    std_acc = np.std(acc_list, ddof=1)
    var_acc = np.var(acc_list)
    print("avg{} var{} std{}".format(avg_acc, std_acc, var_acc))
    print('best_random_team best_acc{}'.format(best_team, best_acc))
    space.append(best_acc)
    print('space:{}'.format(space))
    return new_net_id[best_team[0]], new_net_id[best_team[1]]

sort_net_id=['tx2_gpu16@35ms_top1@70.3_finetune@25', 'note8_lat@22ms_top1@70.4_finetune@25', 's7edge_lat@29ms_top1@70.5_finetune@25',
        'cpu_lat@10ms_top1@71.1_finetune@25', 'LG-G8_lat@8ms_top1@71.1_finetune@25', 'pixel1_lat@20ms_top1@71.4_finetune@25',
        'note10_lat@8ms_top1@71.4_finetune@25', 'pixel2_lat@25ms_top1@71.5_finetune@25', 'v100_gpu64@5ms_top1@71.6_finetune@25',
        'cpu_lat@11ms_top1@72.     0_finetune@25', '1080ti_gpu64@12ms_top1@72.6_finetune@25', 'note8_lat@31ms_top1@72.8_finetune@25',
        'tx2_gpu16@47ms_top1@72.9_finetune@25', 'v100_gpu64@6ms_top1@73.0_finetune@25', 'LG-G8_lat@11ms_to     p1@73.0_finetune@25',
        's7edge_lat@41ms_top1@73.1_finetune@25', 'pixel1_lat@28ms_top1@73.3_finetune@25', 'pixel2_lat@35ms_top1@73.4_finetune@25',
        'note10_lat@11ms_top1@73.6_finetune@25', '1080ti_gpu     64@15ms_top1@73.8_finetune@25', 'cpu_lat@15ms_top1@74.6_finetune@25',
        's7edge_lat@58ms_top1@74.7_finetune@25', 'LG-G8_lat@16ms_top1@74.7_finetune@25', 'pixel2_lat@50ms_top1@74.7_finetune@25',
        'note8_lat@49ms_top1@74.9_finetune@25', 'pixel1_lat@40ms_top1@74.9_finetune@25', '1080ti_gpu64@22ms_top1@75.3_finetune@25',
        'v100_gpu64@9ms_top1@75.3_finetune@25', 'tx2_gpu16@80ms_top1@75.4_finetune@25', 'note10_lat@16ms_top1@75.5_finetune@25',
        'cpu_lat@17ms_top1@75.7_finetune@25', 'tx2_gpu16@96ms_top1@75.8_finetune@25', 'pixel2_lat@62ms_top1@75.8_finetune@25',
        'v100_gpu64@11ms_top1@76.1_finetune@25', 'note8_lat@65ms_top1@76.1_finetune@25', 's7edge_lat@88ms_top1@76.3_finetune@25',
        '1080ti_gpu64@27ms_top1@76.4_finetune@25', 'LG-G8_lat@24ms_top1@76.4_finetune@25', 'pixel1_lat@58ms_top1@76.9_finetune@75',
        'pixel1_lat@79ms_top1@78.7_finetune@75', 'flops@389M_top1@79.1_finetune@75', 'note10_lat@41ms_top1@79.3_finetune@75',
        'flops@482M_top1@79.6_finetune@75', 'note10_lat@50ms_top1@79.7_finetune@75', 'pixel1_lat@132ms_top1@79.8_finetune@75',
        'flops@595M_top1@80.0_finetune@75', 'pixel1_lat@143ms_top1@80.1_finetune@75', 'note10_lat@64ms_top1@80.2_finetune@75']
