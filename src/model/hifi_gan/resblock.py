from torch import nn
from torch.nn.utils.parametrizations import weight_norm


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, dilation_sizes=(1, 2), resblock_type="v3"):
        super().__init__()

        self.convs = nn.ModuleList()
        for d in dilation_sizes:
            self.convs.append(
                weight_norm(
                    nn.Conv1d(in_channels, in_channels, kernel_size, padding=(kernel_size - 1) * d // 2, dilation=d)
                )
            )
            if resblock_type != "v3":
                self.convs.append(
                    weight_norm(nn.Conv1d(in_channels, in_channels, kernel_size, padding=(kernel_size - 1) // 2))
                )
        self.convs.apply(init_weights)

        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        for conv in self.convs:
            res = self.act(x)
            res = conv(res)
            x = x + res

        return x
