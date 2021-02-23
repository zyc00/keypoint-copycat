import sys
sys.path.append('./keypoints')
sys.path.append('./configs')

import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam

from data_augments import TpsAndRotate, nop, TpsAndRotate_s
from keypoints.models import keynet
from utils import ResultsLogger
from keypoints.ds import datasets as ds
from config import config

import numpy as np


if __name__ == '__main__':

    args = config()
    torch.device('cpu')
    run_dir = f'data/models/keypoints/{args.model_type}/run_{args.run_id}'

    """ logging """
    display = ResultsLogger(run_dir=run_dir,
                            num_keypoints=args.model_keypoints,
                            title='Results',
                            visuals=args.display,
                            image_capture_freq=args.display_freq,
                            kp_rows=args.display_kp_rows,
                            comment=args.comment)
    display.header(args)

    """ dataset """
    datapack = ds.datasets[args.dataset]
    train, test = datapack.make(args.dataset_train_len, args.dataset_test_len, data_root=args.data_root)
    pin_memory = False if args.device == 'cpu' else True
    train_l = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=pin_memory, num_workers=)
    test_l = DataLoader(test, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=pin_memory)

    """ data augmentation """
    if args.data_aug_type == 'tps_and_rotate':
        augment = TpsAndRotate(args.data_aug_tps_cntl_pts, args.data_aug_tps_variance, args.data_aug_max_rotate)
    else:
        augment = nop

    """ model """
    kp_network = keynet.make(args).to(args.device)

    """ optimizer """
    optim = Adam(kp_network.parameters(), lr=1e-4)

    """ apex mixed precision """
    # if args.device != 'cpu':
    #     model, optimizer = amp.initialize(kp_network, optim, opt_level=args.opt_level)

    """ loss function """
    def l2_reconstruction_loss(x, x_, k, k_s, loss_mask=None):
        loss = (x - x_) ** 2
        if loss_mask is not None:
            loss = loss * loss_mask
        k_t = eqv_loss(k_s)
        loss_e = (k - k_t) ** 2
        return torch.mean(loss) + torch.mean(loss_e)

    def eqv_loss(k):
        list = np.zeros([16, 10, 128, 128])
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                k_x = k[i][j][0] * 127
                k_y = k[i][j][1] * 127
                k_x = k_x.detach().numpy()
                k_y = k_y.detach().numpy()
                list[i][j][int(k_x)][int(k_y)] = 1
        k_map = torch.from_numpy(list)
        k_map = tuple([k_map, k_map])
        k_, k_map_t, _ = augment(*k_map)
        k_t = torch.zeros([16, 10, 2])
        for i in range(16):
            for j in range(10):
                s = 0
                s_x = 0
                s_y = 0
                for h in range(128):
                    for n in range(128):
                        s_x = s_x + h * k_map_t[i][j][h][n]
                        s_y = s_y + n * k_map_t[i][j][h][n]
                        s = s + k_map_t[i][j][h][n]
                k_t[i][j][0] = s_x / (127 * (s + 0.0001))
                k_t[i][j][1] = s_y / (127 * (s + 0.0001))
        return k_t



    criterion = l2_reconstruction_loss

    def to_device(data, device):
        return tuple([x.to(device) for x in data])

    for epoch in range(1, args.epochs + 1):

        if not args.demo:
            """ training """
            batch = tqdm(train_l, total=len(train) // args.batch_size)
            for i, data in enumerate(batch):
                data = to_device(data, device=args.device)
                x, x_, loss_mask = augment(*data)

                optim.zero_grad()
                x_t, z, k, m, p, heatmap, k_s = kp_network(x, x_)

                loss = criterion(x_t, x_, k, k_s, loss_mask)


                loss.backward()
                optim.step()

                if i % args.checkpoint_freq == 0:
                    kp_network.save(run_dir + '/checkpoint')

                display.log(batch, epoch, i, loss, optim, x, x_, x_t, heatmap, k, m, p, loss_mask, type='train', depth=20)

        """ test  """
        with torch.no_grad():
            batch = tqdm(test_l, total=len(test) // args.batch_size)
            for i, data in enumerate(batch):
                data = to_device(data, device=args.device)
                x, x_, loss_mask = augment(*data)

                x_t, z, k, m, p, heatmap, k_s = kp_network(x, x_)
                loss = criterion(x_t, x_, k, k_s, loss_mask)

                display.log(batch, epoch, i, loss, optim, x, x_, x_t, heatmap, k, m, p, loss_mask, type='test', depth=20)

            ave_loss, best_loss = display.end_epoch(epoch, optim)

            """ save if model improved """
            if ave_loss <= best_loss and not args.demo:
                kp_network.save(run_dir + '/best')