from aihwkit.nn.conversion import convert_to_analog
import os
from torch import nn, save
from models import SRCNN
from aihwkit.simulator.configs import InferenceRPUConfig


model = SRCNN()
rpu_config = InferenceRPUConfig()

model = convert_to_analog(model, rpu_config)

save(model.state_dict(), "basic_conversion.pth")