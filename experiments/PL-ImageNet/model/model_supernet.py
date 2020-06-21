from .fbnet_builder import *

class SampleNet(nn.Module):
    def __init__(self):
        super(SampleNet, self).__init__()
        self.first = ConvBNRelu(input_depth=3, output_depth=16, kernel=3, stride=1,
                                pad=1, no_bias=1, use_relu="relu", bn_type="bn")
        self.branch2_to_search = nn.Sequential(
                                    skip(16, 16, -999, 1),
                                    ir_k3_s2(16, 32, -999, 2),
                                    ir_k5_e1(32, 32, -999, 2),
                                    ir_k5_e1(32, 64, -999, 2)
                                )

        self.branch3_to_search = nn.Sequential(
                                    skip(16, 16, -999, 1),
                                    ir_k3_e3(16, 32, -999, 2),
                                    ir_k5_e1(32, 32, -999, 2),
                                    ir_k3_e1(32, 64, -999, 2),
                                    skip(64, 64, -999, 2),
                                    ir_k5_s2(64, 128, -999, 1)
                                )

        self.last_stages = nn.Sequential(
            ConvBNRelu(input_depth=128+64, output_depth=64, kernel=3, stride=2,
                       pad=1, no_bias=1, use_relu="relu", bn_type="bn"),
            ConvBNRelu(input_depth=64, output_depth=1504, kernel=3, stride=2,
                       pad=1, no_bias=1, use_relu="relu", bn_type="bn"),
            nn.AvgPool2d(kernel_size=7),
            Flatten(),
            nn.Linear(in_features=1504, out_features=1000)
        )
    
    def forward(self, x):
        y2 = y3 = self.first(x)
        y2 = self.branch2_to_search(y2)
        y3 = self.branch3_to_search(y3)
        y3 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(y3)
        y = torch.cat([y2,y3], dim=1)
        y = self.last_stages(y)
        return torch.abs(y)


class SequenceNet(nn.Module):
    def __init__(self):
        super(SequenceNet, self).__init__()
        self.first = ConvBNRelu(input_depth=3, output_depth=16, kernel=3, stride=1,
                                pad=1, no_bias=1, use_relu="relu", bn_type="bn")

        self.branch3_to_search = nn.Sequential(
            ir_k5_e1(16, 16, -999, 1),
            ir_k3_e3(16, 32, -999, 2),
            ir_k3_e3(32, 64, -999, 1),
            ir_k3_e3(64, 64, -999, 1),
            ir_k5_e1(64, 64, -999, 2),
            ir_k5_e1(64, 128, -999, 1),
            ir_k5_e1(128, 128, -999, 1),
            ir_k3_e1(128, 128, -999, 2),
            ir_k3_e1(128, 128, -999, 1),
            ir_k3_e1(128, 256, -999, 1),
            skip(256, 256, -999, 1),
            ir_k5_e1(256, 256, -999, 1),
            ir_k5_e1(256, 256, -999, 1),
            ir_k5_s2(256, 256, -999, 1)
        )

        self.last_stages = nn.Sequential(
            ConvBNRelu(input_depth=256, output_depth=64, kernel=3, stride=2,
                       pad=1, no_bias=1, use_relu="relu", bn_type="bn"),
            ConvBNRelu(input_depth=64, output_depth=1504, kernel=3, stride=2,
                       pad=1, no_bias=1, use_relu="relu", bn_type="bn"),
            nn.AvgPool2d(kernel_size=7),
            Flatten(),
            nn.Linear(in_features=1504, out_features=1000)
        )

    def forward(self, x):
        y = self.first(x)
        y = self.branch3_to_search(y)
        y = self.last_stages(y)
        return torch.abs(y)

