import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from analog_models import PCM_SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr
from torch.utils.data.dataloader import DataLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = PCM_SRCNN()
    model.load_state_dict(torch.load(args.weights_file, map_location=device))
    model.to(device)

    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

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

    print('eval psnr: {:.2f}'.format(epoch_psnr.avg))