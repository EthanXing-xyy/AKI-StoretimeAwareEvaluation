# %%
import torch
import torch.nn as nn
import pytorch_lightning as pl
# %%

class RMSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-8):
        super().__init__()
        self.scale = num_features ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        norm = torch.norm(x, dim=1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g.view(1, -1, 1)

class Residual_Block(pl.LightningModule):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: int,
        stride: int = 1,
        depth_wise: bool = False
    ) -> None:
        super().__init__()

        self.conv_1_1 = nn.Conv1d(
            kernel_size=1,
            stride=stride,
            in_channels=input_channels,
            out_channels=output_channels
        )

        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=hidden_channels,
                kernel_size=13,
                stride=stride,
                padding=6,
            ),
            RMSNorm(
                num_features=hidden_channels
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=output_channels,
                kernel_size=13,
                stride=1,
                padding=6,
            ),
            RMSNorm(
                num_features=output_channels
            ),
            nn.GELU()
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        y_out = self.conv_layers(x) + self.conv_1_1(x)

        return y_out


class Residual_CNN_Model(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        seq_len: int,
    ) -> None:
        super().__init__()

        self.cnn_block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=7
            ),
            RMSNorm(
                64
            ),
            nn.GELU(),
            nn.MaxPool1d(
                kernel_size=5,
                stride=2
            )
        )

        self.res_block_1 = nn.ModuleList(
            [
                Residual_Block(
                    input_channels=64*1,
                    output_channels=64*1,
                    hidden_channels=64*1,
                    stride=1
                ) for i in range(4)
            ]
        )
        self.res_block_2 = nn.ModuleList(
            [
                Residual_Block(
                    input_channels=64*1,
                    output_channels=64*2,
                    hidden_channels=64*2,
                    stride=2
                )
            ]
            +
            [
                Residual_Block(
                    input_channels=64*2,
                    output_channels=64*2,
                    hidden_channels=64*2,
                    stride=1
                ) for i in range(3)
            ]
        )
        self.res_block_3 = nn.ModuleList(
            [
                Residual_Block(
                    input_channels=64*2,
                    output_channels=64*3,
                    hidden_channels=64*3,
                    stride=2
                )
            ]
            +
            [
                Residual_Block(
                    input_channels=64*3,
                    output_channels=64*3,
                    hidden_channels=64*3,
                    stride=1
                ) for i in range(5)
            ]
        )
        self.res_block_4 = nn.ModuleList(
            [
                Residual_Block(
                    input_channels=64*3,
                    output_channels=64*4,
                    hidden_channels=64*4,
                    stride=2
                )
            ]
            +
            [
                Residual_Block(
                    input_channels=64*4,
                    output_channels=64*4,
                    hidden_channels=64*4,
                    stride=1
                ) for i in range(2)
            ]
        )

        # self.linear = nn.Sequential(
        #     nn.Linear(256, output_class),
        # )

        self.global_pool = nn.AvgPool1d(kernel_size=4 if seq_len == 48 else 12)
        
        self.encoder = nn.Sequential(
            self.cnn_block,
            *self.res_block_1,
            *self.res_block_2,
            *self.res_block_3,
            *self.res_block_4,
            self.global_pool,
            nn.Flatten()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        y_embedded = self.encoder(x)

        return y_embedded


class Timeseries_CNN_Model(pl.LightningModule):
    def __init__(self, num_demo_features: int, num_timeseries_features: int, num_classes: int, seq_len: int) -> None:
        super().__init__()

        self.timeseries_cnn = Residual_CNN_Model(in_channels=num_timeseries_features, seq_len=seq_len)
        
        self.linear = nn.Sequential(
            nn.Linear(256 + num_demo_features, 256 + num_demo_features),
            nn.GELU(),
            nn.Linear(256 + num_demo_features, 256 + num_demo_features),
            nn.GELU(),
            nn.Linear(256 + num_demo_features, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x_demo: torch.Tensor, x_time_series: torch.Tensor) -> torch.Tensor:
        
        y_embedded = self.timeseries_cnn(x_time_series.permute(0, 2, 1))
        
        y_concat = torch.cat([y_embedded, x_demo], dim=1)
        
        y_out = self.linear(y_concat)
        
        return y_out

# %%
if __name__ == "__main__":
    
    time_series_data = torch.rand(size=(3, 256, 49))
    
    demo_data = torch.rand(size=(3, 3))
    
    model = Timeseries_CNN_Model(num_demo_features=3, num_timeseries_features=49, num_classes=2, seq_len=256)
    
    output = model.forward(demo_data, time_series_data)
    
    print(output.shape)
    
# %%
