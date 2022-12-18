from torch import nn
from aihwkit.nn import AnalogConv2d
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import WeightNoiseType
from aihwkit.inference import PCMLikeNoiseModel
from aihwkit.simulator.configs.utils import WeightModifierType, WeightClipType
from aihwkit.nn import AnalogLinear, AnalogSequential



class PCM_SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()

        rpu_config = InferenceRPUConfig()

        # specify additional options of the non-idealities in forward to your liking
        rpu_config.forward.inp_res = 1 / 64.  # 6-bit DAC discretization.
        rpu_config.forward.out_res = 1 / 256.  # 8-bit ADC discretization.
        rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
        rpu_config.forward.w_noise = 0.02  # Some short-term w-noise.
        rpu_config.forward.out_noise = 0.02  # Some output noise.

        # specify the noise model to be used for inference only
        rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)  # the model described

        #more experimentation
        rpu_config.mapping.digital_bias = True
        rpu_config.mapping.out_scaling_columnwise = False
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.weight_scaling_omega = 1.0
        rpu_config.mapping.weight_scaling_columnwise = False

        rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
        #rpu_config.remap.type = WeightRemapType.CHANNELWISE_SYMMETRIC
        rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN
        rpu_config.clip.sigma = 2.5

        rpu_config.modifier.type = WeightModifierType.ADD_NORMAL
        rpu_config.modifier.std_dev = 0.1

        self.gen = AnalogSequential(
            AnalogConv2d(num_channels, 64, kernel_size=9, padding=9 // 2, rpu_config=rpu_config),
            AnalogConv2d(64, 32, kernel_size=5, padding=5 // 2, rpu_config=rpu_config),
            AnalogConv2d(32, num_channels, kernel_size=5, padding=5 // 2, rpu_config=rpu_config),
            nn.ReLU(inplace=True),
        )
        # self.conv1 = AnalogConv2d(num_channels, 64, kernel_size=9, padding=9 // 2, rpu_config=rpu_config)
        # self.conv2 = AnalogConv2d(64, 32, kernel_size=5, padding=5 // 2, rpu_config=rpu_config)
        # self.conv3 = AnalogConv2d(32, num_channels, kernel_size=5, padding=5 // 2, rpu_config=rpu_config)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        # x = self.conv3(x)
        # return x
        return self.gen(x)