import os
from multiprocessing import Process
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
from torch import nn
from torchvision import transforms
from os import walk

from model.dataset import FlowersDataset
from model.model import CycleGANModel

torch.manual_seed(0)

def prep_dataset(path_A, path_B):
    load_shape = 260
    target_shape = 256

    transform_A = transforms.Compose([
        transforms.Resize(load_shape),
        transforms.CenterCrop(target_shape),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    transform_B = transform_A

    files_B = []
    for (dirpath, dirnames, filenames) in walk(path_B):
        for filename in filenames:
            files_B.append(os.path.join(dirpath,filename))

    files_A = []
    for (dirpath, dirnames, filenames) in walk(path_A):
        for filename in filenames:
            files_A.append(os.path.join(dirpath,filename))

    dataset = FlowersDataset(files_A,
                             files_B,
                             transform_A=transform_A,
                             transform_B=transform_B)
    return dataset

def train(path_A, path_B, output_path, num_epochs=20):

    dataset = prep_dataset(path_A, path_B)

    adv_criterion = nn.MSELoss()
    recon_criterion = nn.L1Loss()

    lr = 0.0001
    dim_A = 3
    dim_B = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1

    model = CycleGANModel('cuda', output_path)

    model.train(dataset, num_epochs, lr, batch_size, True)

def train_in_process(path_A, path_B, output_path, num_epochs=20):
    p = Process(target=train, args=(path_A, path_B, output_path, num_epochs,))
    p.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Train CycleGAN',
                                    description='Train CycleGAN')
    parser.add_argument('path_A', type=str, help='path to A style images')
    parser.add_argument('path_B', type=str, help='path to B style images')
    parser.add_argument('output_path', type=str, help='path to store weights')
    args = parser.parse_args()
    
    assert os.path.exists(args.path_A), 'check the style A images path!'
    assert os.path.exists(args.path_B), 'check the style B images path!'
    assert os.path.exists(args.output_path), 'check the output_path!'
    
    train(args.path_A, args.path_B, args.output_path)