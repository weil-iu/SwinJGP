import torch
from utils import count_percentage, PSNR, SSIM, MSE

def EVAL(model, data_loader, device, config, epoch=0):
    model.eval()
    mse_total = 0
    psnr_total = 0
    ssim_total = 0
    total = 0
    iconst = model.imod.weight.data
    qconst = model.qmod.weight.data
    
    z_total = torch.zeros((10000, config.channel_use * 2))
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        total += len(target)

        with torch.no_grad():
            _, code, rec = model(data)
            if config.mode == 'test':
                if batch_idx <= int(10000/config.batch_size)-1:
                    z_total[batch_idx * config.batch_size:(batch_idx + 1) * config.batch_size, :] = code
                elif batch_idx == int(10000/config.batch_size):
                    z_total[int(10000/config.batch_size) * config.batch_size:, :] = code
                    count_percentage(z_total, config.mod_method, -1, config.snr_train, config.channel_use, iconst, qconst, config)
            else:
                if batch_idx == 0:
                    count_percentage(code, config.mod_method, epoch, config.snr_train, config.channel_use, iconst, qconst, config)

        mse = MSE(data, rec)
        psnr = PSNR(data, rec)
        ssim = SSIM(data, rec)

        mse_total += mse
        psnr_total += psnr
        ssim_total += ssim

    mse_total /= total
    psnr_total /= total
    ssim_total /= total

    return mse_total, psnr_total, ssim_total


