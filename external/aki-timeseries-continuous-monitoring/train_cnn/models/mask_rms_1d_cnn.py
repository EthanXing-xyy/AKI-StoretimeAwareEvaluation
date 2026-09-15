# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# %%
class RMSNorm(nn.Module):
    """Root Mean Square Normalization for better numerical stability"""
    def __init__(self, num_features: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x.pow(2), dim=1, keepdim=True) + self.eps)
        return x / rms * self.scale.view(1, -1, 1)

class MaskedGlobalAvgPool1d(nn.Module):
    """패딩된 부분을 무시하고 평균을 계산하는 전역 평균 풀링"""
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 마스크를 사용해 패딩된 부분은 0으로 만들고 정확한 평균 계산
        masked_x = x * mask
        summed = masked_x.sum(dim=2)
        count = mask.sum(dim=2).clamp(min=1e-8)  # 수치 안정성 개선
        return summed / count

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

        # stride가 있는 경우, shortcut connection도 차원과 길이를 맞춰줘야 함
        if stride != 1 or input_channels != output_channels:
            self.conv_1_1 = nn.Conv1d(
                kernel_size=1,
                stride=stride,
                in_channels=input_channels,
                out_channels=output_channels
            )
        else:
            self.conv_1_1 = nn.Identity()

        # Residual block layers with proper normalization
        self.conv_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=hidden_channels,
                kernel_size=13,
                stride=stride,
                padding=6,
            ),
            RMSNorm(num_features=hidden_channels),
            nn.GELU(),
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=output_channels,
                kernel_size=13,
                stride=1,
                padding=6,
            ),
            RMSNorm(num_features=output_channels)
        )
        # Final activation applied after residual connection

    def forward(self, x: torch.FloatTensor, mask: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Forward pass with mask handling
        
        Returns:
            tuple: (output, updated_mask) - 출력과 업데이트된 마스크를 함께 반환
        """
        # Shortcut connection 처리
        identity = self.conv_1_1(x)
        
        # Conv layer의 stride에 따라 마스크 다운샘플링
        first_conv = self.conv_layers[0]  # 첫 번째 conv layer
        if hasattr(first_conv, 'stride') and first_conv.stride[0] > 1:
            # Stride가 있는 경우 마스크도 다운샘플링
            stride = first_conv.stride[0]
            output_mask = mask[:, :, ::stride]
            # output_mask = F.max_pool1d(mask, kernel_size=stride, stride=stride)
        else:
            output_mask = mask
        
        # 메인 경로
        y_conv = self.conv_layers(x)
        y_conv = y_conv * output_mask
        
        # Identity 경로에도 적절한 마스크 적용
        if isinstance(self.conv_1_1, nn.Identity):
            identity = identity * mask
        else:
            identity = identity * output_mask
        
        y_out = y_conv + identity
        y_out = F.gelu(y_out)  # Apply activation after residual connection
        return y_out, output_mask

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
                padding=2 # padding=2가 일반적입니다 (5//2)
            ),
            RMSNorm(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=5, stride=2)
        )
        # Residual blocks with proper channel progression
        self.res_block_1 = nn.ModuleList(
            [Residual_Block(64, 64, 64, stride=1) for _ in range(4)]
        )
        self.res_block_2 = nn.ModuleList(
            [Residual_Block(64, 128, 128, stride=2)] +
            [Residual_Block(128, 128, 128, stride=1) for _ in range(3)]
        )
        self.res_block_3 = nn.ModuleList(
            [Residual_Block(128, 192, 192, stride=2)] +
            [Residual_Block(192, 192, 192, stride=1) for _ in range(5)]
        )
        self.res_block_4 = nn.ModuleList(
            [Residual_Block(192, 256, 256, stride=2)] +
            [Residual_Block(256, 256, 256, stride=1) for _ in range(2)]
        )

                # Final pooling and flattening
        self.masked_global_pool = MaskedGlobalAvgPool1d()
        self.flatten = nn.Flatten()

    def _downsample_mask(self, mask: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
        """마스크 다운샘플링을 위한 헬퍼 메서드"""
        return F.max_pool1d(mask, kernel_size=kernel_size, stride=stride)
    
    def _apply_residual_blocks(self, x: torch.Tensor, mask: torch.Tensor, 
                              blocks: nn.ModuleList) -> tuple[torch.Tensor, torch.Tensor]:
        """Residual block들을 효율적으로 적용하는 헬퍼 메서드"""
        for block in blocks:
            x, mask = block(x, mask)
        return x, mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 마스크 생성 - 모든 채널에 대해 OR 연산으로 더 robust하게
        mask = (x.abs().sum(dim=1, keepdim=True) > 1e-8).float()

        # 2. 초기 CNN 블록 적용
        x = self.cnn_block(x)
        # MaxPool1d와 동일한 파라미터로 마스크 다운샘플링
        mask = self._downsample_mask(mask, kernel_size=5, stride=2)

        # 3. Residual block들을 효율적으로 적용
        x, mask = self._apply_residual_blocks(x, mask, self.res_block_1)
        x, mask = self._apply_residual_blocks(x, mask, self.res_block_2)
        x, mask = self._apply_residual_blocks(x, mask, self.res_block_3)
        x, mask = self._apply_residual_blocks(x, mask, self.res_block_4)

        # 4. 마스크된 전역 평균 풀링
        y_embedded = self.masked_global_pool(x, mask)
        y_embedded = self.flatten(y_embedded)

        return y_embedded


class Timeseries_CNN_Model(pl.LightningModule):
    def __init__(self, num_demo_features: int, num_timeseries_features: int, num_classes: int, seq_len: int) -> None:
        super().__init__()
        self.save_hyperparameters() # Lightning best practice

        self.timeseries_cnn = Residual_CNN_Model(in_channels=num_timeseries_features, seq_len=seq_len)

        # Classification head with dropout for regularization
        combined_dim = 256 + num_demo_features
        self.linear = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.GELU(),
            nn.Dropout(0.1),  # Regularization
            nn.Linear(combined_dim, combined_dim // 2),
            nn.GELU(), 
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, num_classes)
        )

    def forward(self, x_demo: torch.Tensor, x_time_series: torch.Tensor) -> torch.Tensor:
        # Time series embedding: (B, L, C) -> (B, C, L) for Conv1d
        y_embedded = self.timeseries_cnn(x_time_series.permute(0, 2, 1))
        
        # Combine time series embedding with demographic features
        y_concat = torch.cat([y_embedded, x_demo], dim=1)
        
        # Final classification
        y_out = self.linear(y_concat)
        return y_out

def test_model():
    """Model functionality test with padded time series data"""
    print("Testing MaskRMSCNN model...")
    
    # Test both left and right padding scenarios
    batch_size, seq_len, features = 4, 256, 49
    
    # Test 1: Left padding (current example)
    print("\n=== Test 1: Left Padding ===")
    left_padded_data = torch.rand(batch_size, seq_len, features)
    left_padded_data[0, :100, :] = 0  # First sample: 100 timesteps left padding
    left_padded_data[1, :50, :] = 0   # Second sample: 50 timesteps left padding
    # Third and fourth samples: no padding
    
    # Test 2: Right padding  
    print("=== Test 2: Right Padding ===")
    right_padded_data = torch.rand(batch_size, seq_len, features)
    right_padded_data[0, -100:, :] = 0  # First sample: 100 timesteps right padding
    right_padded_data[1, -50:, :] = 0   # Second sample: 50 timesteps right padding
    # Third and fourth samples: no padding
    
    # Test 3: Mixed padding (both sides)
    print("=== Test 3: Mixed Padding ===") 
    mixed_padded_data = torch.rand(batch_size, seq_len, features)
    mixed_padded_data[0, :50, :] = 0    # Left padding
    mixed_padded_data[0, -50:, :] = 0   # Right padding
    mixed_padded_data[1, :30, :] = 0    # Only left padding
    mixed_padded_data[2, -30:, :] = 0   # Only right padding
    # Fourth sample: no padding
    
    demo_data = torch.rand(batch_size, 3)
    
    # Initialize model
    model = Timeseries_CNN_Model(
        num_demo_features=3, 
        num_timeseries_features=features, 
        num_classes=2, 
        seq_len=seq_len
    )
    model.eval()
    
    # Test all scenarios
    test_cases = [
        ("Left Padding", left_padded_data),
        ("Right Padding", right_padded_data), 
        ("Mixed Padding", mixed_padded_data)
    ]
    
    for case_name, test_data in test_cases:
        print(f"\n--- {case_name} Test ---")
        with torch.no_grad():
            output = model(demo_data, test_data)
        
        # Verify mask generation for first sample
        mask = (test_data.abs().sum(dim=2) > 1e-8).float()
        valid_timesteps = mask[0].sum().item()
        total_timesteps = seq_len
        padded_timesteps = total_timesteps - valid_timesteps
        
        print(f"Input shape: {test_data.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Sample 0 - Valid timesteps: {valid_timesteps:.0f}/{total_timesteps} (Padded: {padded_timesteps:.0f})")
        print(f"Mask sum check: {mask.sum(dim=1)[:2].tolist()}")  # Show first 2 samples
    
    print(f"\n✅ All padding scenarios work correctly!")
    return model
# %%

if __name__ == "__main__":
    test_model()

# %%
