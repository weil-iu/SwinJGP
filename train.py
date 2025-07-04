import os
import torch
from torch import optim, nn
import time
from evaluation import EVAL
from utils import save_checkpoint, PSNR


def train(config, net, train_iter, test_iter, device):
    learning_rate = config.lr
    epochs = config.train_iters
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    mse_f = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.train_iters+1, T_mult=1, eta_min=1e-6, last_epoch=-1)

    for epoch in range(epochs):
        start = time.time()
        net.train()
        epoch_loss = []
        psnr_total_train = 0

        for i, (X, Y) in enumerate(train_iter):
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            KL, _, y_recon = net(X)

            mse = mse_f(y_recon, X) * 16256.25 #(255/2)^2
            if config.Training_strategy == 'Proposed':
                loss = mse + config.KLlambda * KL
            elif config.Training_strategy == 'QAM':
                loss = mse
                
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.cpu().item())
            psnr = PSNR(X, y_recon.detach())
            psnr_total_train += psnr

        scheduler.step()

        loss = sum(epoch_loss) / len(epoch_loss)

        mse, psnr, ssim = EVAL(net, test_iter, device, config, epoch)
        print('\n' + '-' * 50)
        print('epoch: {:d}, loss: {:.6f}, KL: {:.3f}, mse: {:.6f}, psnr: {:.3f}, ssim: {:.3f}, lr: {:.6f}'.format
              (epoch, loss, KL, mse, psnr, ssim, optimizer.state_dict()['param_groups'][0]['lr']))

        # if (epochs - epoch) <= 10:
        if (epochs - epoch) <= 10:    
            file_name = config.model_path + '/{}/'.format(config.mod_method)
            if not os.path.exists(file_name):
                os.makedirs(file_name)
            model_name = config.Training_strategy + '_' + config.channel + '_SNR{:.3f}_Trans{:d}_{}.pth.tar'.format(
                config.snr_train, config.channel_use, config.mod_method)
            save_checkpoint(net.state_dict(), file_name + model_name)
        
        spendTime = time.time() - start
        print(f"time : {spendTime:}s")



