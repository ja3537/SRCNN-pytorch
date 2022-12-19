import argparse
import os
import copy

import torch
from torch import nn, save
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from analog_models import PCM_SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr
import pickle


from aihwkit.inference import PCMLikeNoiseModel
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import InferenceRPUConfig
# from aihwkit.simulator.configs.utils import WeightNoiseType
from aihwkit.simulator.rpu_base import cuda
from aihwkit.nn.conversion import convert_to_analog


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--model-file', type=str, default='trained_models/srcnn_x4.pth')
    parser.add_argument('--prev-epoch', type=int, required=True)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.manual_seed(args.seed)

    rpu_config = InferenceRPUConfig()
    rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)

    model = SRCNN()
    model.load_state_dict(torch.load(args.model_file, map_location=device))

    model = PCM_SRCNN()
    model_file= ''
    model.load_state_dict(torch.load( os.path.join(args.outputs_dir, 'best_{}_{}_{}.pth'.format(args.lr, args.prev_epoch, args.scale))), map_location=device))

    with open("trained_models/losses.pkl", 'rb') as f:
        losses = pickle.load(f)
    with open("trained_models/eval_psnr.pkl", 'rb') as f:
        eval_psnr = pickle.load(f)
    # #set rpuconfig for PCM and convert model
    # rpu_config = InferenceRPUConfig()
    #
    # # specify additional options of the non-idealities in forward to your liking
    # rpu_config.forward.inp_res = 1 / 64.  # 6-bit DAC discretization.
    # rpu_config.forward.out_res = 1 / 256.  # 8-bit ADC discretization.
    # rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
    # rpu_config.forward.w_noise = 0.02  # Some short-term w-noise.
    # rpu_config.forward.out_noise = 0.02  # Some output noise.
    #
    # # specify the noise model to be used for inference only
    # rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)  # the model described
    #
    model = convert_to_analog(model, rpu_config)

    if cuda.is_compiled():
        model.cuda()
    else:
        model.cpu()
    print('if cuda is compiled:')
    print(cuda.is_compiled())
    criterion = nn.MSELoss()

    #optimizer = optim.SGD(model.parameters(), lr=0.05)
    optimizer = AnalogSGD(model.parameters(), lr=args.lr)
    #optimizer.regroup_param_groups(model)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    #best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()
        total_loss = 0

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to('cpu')
                labels = labels.to('cpu')

                preds = model(inputs)

                loss = criterion(preds, labels)
                total_loss += loss

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        #torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))
        losses.append(total_loss / len(inputs))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            psnr = calc_psnr(preds, labels)
            epoch_psnr.update(psnr, len(inputs))
            eval_psnr.append(psnr / len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            save(model.state_dict(),
                       os.path.join(args.outputs_dir, 'best_{}_{}_{}.pth'.format(args.lr, epoch + args.prev_epoch, args.scale)))
            with open("trained_models/losses.pkl", 'wb') as f:
                pickle.dump(losses, f)
            with open("trained_models/eval_psnr.pkl", 'wb') as f:
                pickle.dump(eval_psnr, f)
            #best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    #torch.save(best_weights, os.path.join(args.outputs_dir, 'best_{}_{}_{}.pth'.format(args.lr, args.num_epochs, args.scale)))
