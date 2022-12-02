import datetime
import os
import time
import torch
import numpy as np
from torch import nn, optim
from torch.optim import lr_scheduler

from option import opt
from datasets import Train_data
from torch.utils.data import DataLoader
from build_models import Image_VSR, initialize_weights
from torch.autograd import Variable
from loss_functions import PSNR, SSIM
from test import test

systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
def main():
    torch.manual_seed(opt.seed)
    train_data = Train_data()
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=opt.threads, pin_memory=False, drop_last=False)

    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else:
        print('--------------------Exist cuda--------------------')
        use_gpu = torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True

    model = Image_VSR()
    print(model)
    model.apply(initialize_weights)
    args = sum(p.numel() for p in model.parameters()) / 1000000
    print('args=', args)
    with open('./results/parameter.txt', 'a+') as f:
        f.write('Parma = {:.6f}M'.format(args) + '\n')
    criterion = nn.L1Loss()

    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)

    for epochs in range(opt.start_epoch + 1, opt.num_epochs + 1):
        for steps, data in enumerate(train_dataloader):
            start_time = time.time()
            inputs, labels = data
            if use_gpu:
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda()

            outputs = model(inputs)
            loss_mse = criterion(labels, outputs)
            optimizer.zero_grad()
            loss_mse.backward()

            nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optimizer.step()
            end_time = time.time()
            cost_time = end_time - start_time
            if steps % 30 == 0:
                print('===> Epochs[{}]({}/{}) || Time = {:.3f}s'.format(epochs, steps+1, len(train_dataloader), cost_time),
                      'loss_mse = {:.8f}'.format(loss_mse))
        scheduler.step()

        if epochs % 10 == 0:
            save_models(model, epochs)


def save_models(model, epochs):
    save_model_path = os.path.join(opt.save_model_path, systime)
    if not os.path.exists(save_model_path):
        os.makedirs(os.path.join(save_model_path))
    save_name = 'X' + str(opt.scale) + '_epoch_{}.pth'.format(epochs)
    checkpoint = {"net": model.state_dict()}
    torch.save(checkpoint, os.path.join(save_model_path, save_name))
    print('Checkpoints save to {}'.format(save_model_path))


if __name__ == '__main__':
    main()
