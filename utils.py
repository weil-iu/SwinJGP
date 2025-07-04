import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as comp_psnr
from skimage.metrics import structural_similarity as comp_ssim


def init_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 42:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=>Saving checkpoint")
    torch.save(state, filename)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=>Saving checkpoint")
    torch.save(state, filename)


def count_percentage(code, mod, epoch, snr, channel_use, iconst, qconst, config):
    N = code.size(1)
    half = N//2
    code = torch.stack([code[:, :half], code[:, half:]], dim=-1).cpu()
    code = code.reshape(-1, 2).cpu()

    if mod == '64qam':
        order = 64

    map = torch.stack([iconst, qconst], dim=-1).reshape(order, 2).cpu()
    per_s = []
    for i in range(order):
        temp = torch.sum(torch.abs(code - map[i, :]), dim=1)
        num = code.shape[0] - torch.count_nonzero(temp).item()
        per = num / code.shape[0]
        per_s.append(per)
    per_s = torch.tensor(per_s).cpu()
    file_name = './cons_fig/' + config.channel + '/' + '{}_{}_{}_'.format(mod, snr, channel_use) + config.Training_strategy
    if not os.path.exists(file_name):
        os.makedirs(file_name)

    fig = plt.figure(dpi=300)
    for k in range(order):
        plt.scatter(map[k, 0], map[k, 1], s=1000 * per_s[k], color='b')
    plt.xlabel('I')
    plt.ylabel('Q')
    fig.savefig(file_name + '/scatter_{}'.format(epoch))
    plt.close()

def PSNR(tensor_org, tensor_trans):
    total_psnr = 0
    origin = ((tensor_org + 1) / 2).cpu().numpy()
    trans = ((tensor_trans + 1) / 2).cpu().numpy()
    for i in range(np.size(trans, 0)):
        psnr = 0
        for j in range(np.size(trans, 1)):
            psnr_temp = comp_psnr(origin[i, j, :, :], trans[i, j, :, :])
            psnr = psnr + psnr_temp
        psnr /= 3
        total_psnr += psnr
    return total_psnr


def SSIM(tensor_org, tensor_trans):
    total_ssim = 0
    origin = ((tensor_org + 1) / 2).cpu().numpy()
    trans = ((tensor_trans + 1) / 2).cpu().numpy()
    for i in range(np.size(trans, 0)):
        ssim = 0
        for j in range(np.size(trans, 1)):
            ssim_temp = comp_ssim(origin[i, j, :, :], trans[i, j, :, :], data_range=1.0)
            ssim = ssim + ssim_temp
        ssim /= 3
        total_ssim += ssim
    return total_ssim


def MSE(tensor_org, tensor_trans):
    origin = ((tensor_org + 1) / 2).cpu().numpy()
    trans = ((tensor_trans + 1) / 2).cpu().numpy()
    mse = np.mean((origin - trans) ** 2)
    return mse * tensor_org.shape[0]


def manual_2d_gaussian_pdf(x1, x2):
    exponent = -0.5 * (x1**2 + x2**2)
    normalization = 1 / (2 * torch.pi)
    return normalization * torch.exp(exponent)


def getActualP(discrete_code):
    B, N, num_classes = discrete_code.shape
    ActualP = discrete_code.sum(dim=(0, 1)) / (B * N)
    return ActualP


def getCaculateP(power_emp, iconst, qconst):
    normalization_iconst = (1.0 / power_emp) ** 0.5 * iconst
    normalization_qconst = (1.0 / power_emp) ** 0.5 * qconst
    fp = manual_2d_gaussian_pdf(normalization_iconst, normalization_qconst)
    CaculateP = fp/torch.sum(fp)
    return CaculateP


def normalize(x, power=1):
    power_emp = torch.mean(x ** 2)
    x = (power / power_emp) ** 0.5 * x
    return power_emp, x


def awgn(snr, x, device):
    # snr(db)
    n = 1.0 / (10.0 ** (snr / 10.0))
    sqrt_n = n ** 0.5
    noise = torch.randn_like(x) * sqrt_n
    noise = noise.to(device)
    x_hat = x + noise
    return x_hat

def rayleigh(input_layer, snr):
    n = 1.0 / (10.0 ** (snr / 10.0))
    std = n ** 0.5
    noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer)).to(input_layer.device)
    noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer)).to(input_layer.device)
    noise = noise_real + 1j * noise_imag
    h = torch.sqrt(torch.normal(mean=0.0, std=1, size=np.shape(input_layer)) ** 2
                    + torch.normal(mean=0.0, std=1, size=np.shape(input_layer)) ** 2) / np.sqrt(2)
    h = h.to(input_layer.device)
    y = input_layer * h + noise
    return y/h

    

