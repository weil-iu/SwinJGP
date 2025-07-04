import torchvision
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
from model import SwinJGP
from train import train
from evaluation import EVAL
from utils import init_seeds
import os
import argparse

def main(config):
    # initialize random seed
    init_seeds()

    # prepare training & test data
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_data = torchvision.datasets.CIFAR10(
        root=config.dataset_path,
        train=True,
        transform=transform_train,
        download=True
    )
    test_data = torchvision.datasets.CIFAR10(
        root=config.dataset_path,
        train=False,
        transform=transform_test,
        download=True
    )

    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=config.batch_size, shuffle=True)

    device = torch.device(config.device)
    net = SwinJGP(config, device).to(device)

    if config.load_checkpoint:
        model_location = ''
        net.load_state_dict(torch.load(model_location, map_location=torch.device('cpu')),strict=False)
        print('loading...' + model_location)

    if config.mode == 'train':
        print('Train Start!')
        train(config, net, train_loader, test_loader, device)

    elif config.mode == 'test':
        print("Test Start!")
        mse, psnr, ssim = EVAL(net, test_loader, device, config, -1)
        print('mse: {:3f}, psnr: {:.3f}, ssmi: {:.3f}'.format(mse, psnr, ssim))

    else:
        print("Wrong mode input!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--load_checkpoint', type=int, default=0)
    parser.add_argument('--mod_method', type=str, default='64qam')
    parser.add_argument('--channel', type=str, choices=['awgn','rayleigh'], default='awgn')
    parser.add_argument('--Training_strategy', type=str, choices=['QAM','Proposed'], default='Proposed')

    # training hyper-parameters
    parser.add_argument('--channel_use', type=int, default=256)
    parser.add_argument('--train_iters', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4) #1e-3
    parser.add_argument('--snr_train', type=float, default=15)
    parser.add_argument('--snr_test', type=float, default=15)
    parser.add_argument('--KLlambda', type=float, default=100.0)

    # misc
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--dataset_path', type=str, default='../dataset/')

    config = parser.parse_args()
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)

    main(config)

