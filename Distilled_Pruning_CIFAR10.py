import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torch import optim
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import copy
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models import resnet50, resnet34, resnet18, wide_resnet50_2, ResNet50_Weights, alexnet
from transformer_models import cait, convnet, simplevit, swin,  vit, vit_small
import gc
import os
import pandas as pd
from torchvision.io import read_image
from flax.training import checkpoints
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

#Import from MTT code
from networks import ConvNet, AlexNet
from distill import ParamDiffAug
from utils import evaluate_synset, evaluate_sourceset, get_network
import argparse

import optuna
import kornia
import wandb

labels_train = torch.load('./data/cifar10_10ipc_labels.pt')
images_train = torch.load('./data/cifar10_10ipc_images.pt')

batch_size = 256
train_dataset = torchvision.datasets.CIFAR10(root = './data',
                                                    train = True,
                                                    transform = transforms.Compose([
                                                            transforms.ToTensor(),
                                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),]),
                                                    download=True)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = True)


test_dataset = torchvision.datasets.CIFAR10(root = './data',
                                                    train = False,
                                                    transform = transforms.Compose([
                                                            transforms.ToTensor(),
                                                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),]),
                                                    download=True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = True)

#Standard train function with hyperparameters used in paper set as default
def train(model,train_loader, num_epochs, lr = .0008, weight_decay = .0008, gamma = .15, milestones = [50,65,80]):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    cost = nn.CrossEntropyLoss()
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma= gamma)
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            loss = cost(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    pass

#Standard test function, prints & returns test accuracy
def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # Test the model
    model.eval()
    model.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader): 
            images, labels = images.to(device), labels.to(device)
            test_output = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            correct += (pred_y == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / total

    print('Test Accuracy:', accuracy)
    return accuracy

#Helper function for prunable all pruning modules to work pytorch global pruning. 
#See global pruning section of this: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
def get_parameters_to_prune(model):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    return tuple(parameters_to_prune)

#Returns number of zeros and total number of prunable parameters of a model. Global Sparsity measured as: zero / total
def sparsity_print(model):
    prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=0)
    zero = total = 0
    for module, _ in get_parameters_to_prune(model):
        zero += float(torch.sum(module.weight == 0))
        total += float(module.weight.nelement())
    print('Number of Zero Weights:', zero)
    print('Total Number of Weights:', total)
    print('Sparsity', zero/total)
    #TODO: Implement Node Sparsity
    return zero, total

def DistilledPruning(model, name, path, images_train, labels_train, train_loader, test_loader, input_args, start_iter = 0, end_iter = 30, num_epochs_distilled = 1000, num_epochs_real = 60, k = 0, amount = .2, save_model = True, validate = False, seed = 0, reinit = False, reinit_model = None, distilled_lr = .01):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    accs = []
    zeros = []
    totals = []
    reinit_acc = []
    time_takens = []
    sparsities = []
    
    if start_iter > 0:
        if os.path.exists(f'{os.getcwd()}/dumps/{"syn" if input_args.distilled_pruning else "source"}/{name}/{seed}/log_{start_iter-1}'):
            [accs, zeros, totals, reinit_acc, time_takens, sparsities] = list(map(list, np.load(f'{os.getcwd()}/dumps/{"syn" if input_args.distilled_pruning else "source"}/{name}/{seed}/log_{start_iter-1}')))
        else:
            print(f'{"-"*20}+error wrong start_iter+{"-"*20}')
    #Create rewind weights at initailization

    if os.path.exists(f'{os.getcwd()}/saves/{"syn" if input_args.distilled_pruning else "source"}/{name}/{seed}/initial_weight.pth'):
        model.load_state_dict(torch.load(f'{os.getcwd()}/saves/{"syn" if input_args.distilled_pruning else "source"}/{name}/{seed}/initial_weight.pth'))
        model_rewind = copy.deepcopy(model).to(device)
    else:
        model_rewind = copy.deepcopy(model).to(device)
        #torch.save(model.state_dict(), path + name + '_RewindWeights' + '_' + str(k))
        if not os.path.exists(f'{os.getcwd()}/saves/{"syn" if input_args.distilled_pruning else "source"}/{name}/{seed}'):
            os.mkdir(f'{os.getcwd()}/saves/{"syn" if input_args.distilled_pruning else "source"}/{name}/{seed}')
        torch.save(model.state_dict(), f'{os.getcwd()}/saves/{"syn" if input_args.distilled_pruning else "source"}/{name}/{seed}/initial_weight.pth') 
    
    #Use if you want to try rewinding to an early point in training, this does not work well, so we suggest k=0 always.
    if k != 0:
        args = argparse.Namespace(lr_net=str(distilled_lr), device='cuda', epoch_eval_train=str(k),batch_train=512,dataset='cifar10',dsa=True,dsa_strategy='color_crop_cutout_flip_scale_rotate',dsa_param = ParamDiffAug(), dc_aug_param=None, zca_trans=kornia.enhance.ZCAWhitening(eps=0.1, compute_inv=True)) #, zca_trans=kornia.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
        model_rewind, acc_train_list, acc_test = evaluate_synset(0, model_rewind,images_train,labels_train,test_loader,args)
        
    for i in range(start_iter,end_iter):
        print('Distilled Pruning Iteration ', i)
        #Set distilled pruning training args for MTT eval
        args = argparse.Namespace(lr_net='.01', device='cuda', epoch_eval_train=str(num_epochs_distilled),batch_train=512,dataset='cifar10',dsa=True,dsa_strategy='color_crop_cutout_flip_scale_rotate',dsa_param = ParamDiffAug(), dc_aug_param=None, zca_trans=kornia.enhance.ZCAWhitening(eps=0.1, compute_inv=True)) #, zca_trans=kornia.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
        #MTT Training on Distilled Data
        if input_args.distilled_pruning:
            model, acc_train_list, acc_test, train_time = evaluate_synset(i, model,images_train,labels_train,test_loader,args)
        else:
            model, acc_train_list, acc_test, train_time = evaluate_sourceset(i, model,train_loader,test_loader,args)
        time_takens.append(train_time)
        prune.global_unstructured(get_parameters_to_prune(model),pruning_method=prune.L1Unstructured,amount=amount)
        #Rewind Weights
        for idx, (module, _) in enumerate(get_parameters_to_prune(model)):
            with torch.no_grad():
                module_rewind = get_parameters_to_prune(model_rewind)[idx][0]
                module.weight_orig.copy_(module_rewind.weight)
    
        if save_model:
            #torch.save(model.state_dict(), path + name + '_iter' + str(i+1))
            torch.save(model.state_dict(), f'{os.getcwd()}/saves/{"syn" if input_args.distilled_pruning else "source"}/{name}/{seed}/weight_{i}.pth')
            
        #Rewind weights back to initialization and train on real data to validate this sparsity mask
        if validate:
            train(model, train_loader,num_epochs = num_epochs_real)
            accs.append(test(model, test_loader))
            zero, total = sparsity_print(model)
            zeros.append(zero)
            totals.append(total)
            sparsities.append(round(zero/total), 3)
            #Rewind Weights
            for idx, (module, _) in enumerate(get_parameters_to_prune(model)):
                with torch.no_grad():
                    module_rewind = get_parameters_to_prune(model_rewind)[idx][0]
                    module.weight_orig.copy_(module_rewind.weight)
            if not os.path.exists(f'{os.getcwd()}/dumps/{"syn" if input_args.distilled_pruning else "source"}/{name}/{seed}'):
                os.mkdir(f'{os.getcwd()}/dumps/{"syn" if input_args.distilled_pruning else "source"}/{name}/{seed}')
            np.save(f'{os.getcwd()}/dumps/{"syn" if input_args.distilled_pruning else "source"}/{name}/{seed}/log_{i}', np.array([accs, zeros, totals, reinit_acc, time_takens, sparsities]))
        
        if reinit:
            #Rewind Weights to Reinit Model
            for idx, (module, _) in enumerate(get_parameters_to_prune(model)):
                with torch.no_grad():
                    module_reinit = get_parameters_to_prune(reinit_model)[idx][0]
                    module.weight_orig.copy_(module_reinit.weight)
                    
            train(model, train_loader,num_epochs = num_epochs_real)
            reinit_acc.append(test(model, test_loader))
            
            for idx, (module, _) in enumerate(get_parameters_to_prune(model)):
                with torch.no_grad():
                    module_rewind = get_parameters_to_prune(model_rewind)[idx][0]
                    module.weight_orig.copy_(module_rewind.weight)
            np.save(path + name + '_log', np.array([accs, zeros, totals, reinit_acc, time_takens, sparsities]))
        else:
            reinit_acc.append(0)
        f = open(f'{os.getcwd()}/logs/{name}/{seed}/seed_logs.txt', 'w')
        f.write(f'{i}')
    #If validate = False, then we still want to validate the final sparsity mask. just not all the masks.
    if not validate:
        train(model, train_loader,num_epochs = num_epochs_real)
        acc = (test(model, test_loader))
        zero, total = sparsity_print(model)
        np.save(path + name + '_log', np.array([acc, zero, total, reinit]))
    
    np.save(f'{os.getcwd()}/dumps/{"syn" if input_args.distilled_pruning else "source"}/{name}/{seed}/time_takens.dat', np.array(time_takens))
    np.save(f'{os.getcwd()}/dumps/{"syn" if input_args.distilled_pruning else "source"}/{name}/{seed}/bestaccuracies.dat', np.array(accs))
    np.save(f'{os.getcwd()}/dumps/{"syn" if input_args.distilled_pruning else "source"}/{name}/{seed}/sparsities.dat', np.array(sparsities))

def main(input_args):
    model = None
    if input_args.model == "cait":
        model = cait.CaiT( 
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 512,
            depth = 6,   # depth of transformer for patch to patch attention only
            cls_depth=2, # depth of cross attention of CLS tokens to patch
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1,
            layer_dropout = 0.05)
    elif input_args.model == "cait_small":
        model = cait.CaiT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 512,
            depth = 6,   # depth of transformer for patch to patch attention only
            cls_depth=2, # depth of cross attention of CLS tokens to patch
            heads = 6,
            mlp_dim = 256,
            dropout = 0.1,
            emb_dropout = 0.1,
            layer_dropout = 0.05)
    elif input_args.model == "swin":
        model = swin.swin_t(
            window_size=4,
            num_classes = 10,
            downscaling_factors=(2,2,2,1))
    elif input_args.model == "simplevit":
        model = simplevit.SimpleViT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 512,
            depth = 6,
            heads = 8,
            mlp_dim = 512)
    elif input_args.model == "vit":
        model = vit.ViT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 512,
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1)
    elif input_args.model == "vit_small":
        model = vit_small.ViT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 512,
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1)
    elif input_args.model == "vit_tiny":
        model = vit_small.ViT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 512,
            depth = 4,
            heads = 6,
            mlp_dim = 256,
            dropout = 0.1,
            emb_dropout = 0.1)
    name = input_args.name
    path = input_args.path
    start_iter = input_args.start_iter
    end_iter = input_args.end_iter
    num_epochs_distilled = input_args.num_epochs_distilled
    num_epochs_real = input_args.num_epochs_real
    k = input_args.k
    amount = input_args.amount
    save_model = input_args.save_model
    validate = input_args.validate
    reinit = input_args.reinit
    reinit_model = input_args.reinit_model
    distilled_lr = input_args.distilled_lr
    for seed in range(input_args.seeds):
        if os.path.exists(f'{os.getcwd()}/logs/{name}/{seed}/seed_logs.txt'):
            f = open(f'{os.getcwd()}/logs/{name}/{seed}/seed_logs.txt', 'r')
            iter_num= f.readline()
            if iter_num+1 == end_iter:
                continue
            elif iter_num > start_iter:
                DistilledPruning(model, name, path, images_train, labels_train, train_loader, test_loader, input_args,
                     start_iter = iter_num+1, end_iter = end_iter, num_epochs_distilled = num_epochs_distilled,
                     num_epochs_real = num_epochs_real, k = k, amount = amount, save_model = save_model,
                     validate = validate, seed = seed, reinit = reinit, reinit_model = reinit_model,
                     distilled_lr = distilled_lr)
            else:
                DistilledPruning(model, name, path, images_train, labels_train, train_loader, test_loader, input_args,
                     start_iter = start_iter, end_iter = end_iter, num_epochs_distilled = num_epochs_distilled,
                     num_epochs_real = num_epochs_real, k = k, amount = amount, save_model = save_model,
                     validate = validate, seed = seed, reinit = reinit, reinit_model = reinit_model,
                     distilled_lr = distilled_lr)
        else:
            DistilledPruning(model, name, path, images_train, labels_train, train_loader, test_loader, input_args,
                     start_iter = start_iter, end_iter = end_iter, num_epochs_distilled = num_epochs_distilled,
                     num_epochs_real = num_epochs_real, k = k, amount = amount, save_model = save_model,
                     validate = validate, seed = seed, reinit = reinit, reinit_model = reinit_model,
                     distilled_lr = distilled_lr)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vit_tiny", type=str)
    parser.add_argument("--name",  default="vit_tiny", type=str)
    parser.add_argument("--path", default=os.getcwd(), type=str)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=30, type=int)
    parser.add_argument("--num_epochs_distilled", default=3000, type=int)
    parser.add_argument("--num_epochs_real", default=80, type=int)
    parser.add_argument("--k", default=0, type=int)
    parser.add_argument("--amount", default=0.2, type=float)
    parser.add_argument("--save_model", default=True, type=bool)
    parser.add_argument("--validate", default=False, type=bool)
    parser.add_argument("--seeds", default=3, type=int)
    parser.add_argument("--reinit", default=False, type=bool)
    parser.add_argument("--reinit_model", default=None, type=str)
    parser.add_argument("--distilled_lr", default=0.01, type=float)
    parser.add_argument("--distilled_pruning", default=False, type=bool)
    # syn_data에 대해 50ipc : lr=0.01, epoch=1000 | 10ipc : lr=0.007, epoch=3000
    # real_data에 대해 lr = 0.008, batch_size=512, weight_dacay=0.0008, gamma=0.15
    input_args = parser.parse_args()

    main(input_args)