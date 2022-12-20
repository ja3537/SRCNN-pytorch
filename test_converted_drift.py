import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import SRCNN
from analog_models import PCM_SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr

from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.nn.conversion import convert_to_analog

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--drift', default=False, action="store_true")
    parser.add_argument('--compensation', default=False, action='store_true')
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN()
    model.load_state_dict(torch.load(args.weights_file, map_location=device))
    rpu_config = InferenceRPUConfig()
    rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    if args.drift and args.compensation:
        rpu_config.drift_compensation = GlobalDriftCompensation()
    model = convert_to_analog(model, rpu_config)
    tmp = PCM_SRCNN(drift_compensation=args.compensation)
    tmp.load_state_dict(model.state_dict())
    model = tmp
    if args.drift:
        model.drift_analog_weights(1000)
    model.to(device)


    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
    image.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    psnr = calc_psnr(y, preds)
    print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace('.', '_srcnn_converted_x{}_drift{}_comp{}.'.format(args.scale, args.drift, args.compensation)))