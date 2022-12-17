from aihwkit.nn.conversion import convert_to_analog
import os
from torch import nn, save
from models import SRCNN
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import WeightNoiseType
from aihwkit.inference import PCMLikeNoiseModel


model = SRCNN()
rpu_config = InferenceRPUConfig()

rpu_config.forward.inp_res = 1 / 64.  # 6-bit DAC discretization.
rpu_config.forward.out_res = 1 / 256.  # 8-bit ADC discretization.
rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
rpu_config.forward.w_noise = 0.02  # Some short-term w-noise.
rpu_config.forward.out_noise = 0.02  # Some output noise.

# specify the noise model to be used for inference only
rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)  # the model described

model = convert_to_analog(model, rpu_config)

save(model.state_dict(), "basic_conversion.pth")
print("saved")