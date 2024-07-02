import torch
import torch.nn as nn

# 定义特征融合的MLP模型
class FeatureFusionMLP(nn.Module):
    def __init__(self, biformer_dim, resnet_dim, hidden_dim, output_dim):
        super(FeatureFusionMLP, self).__init__()
        
        # MLP的输入层
        self.input_dim = biformer_dim + resnet_dim
        
        # 定义MLP的隐藏层
        self.hidden_layer = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 定义MLP的输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, biformer_features, resnet_features):
        # 将Biformer和ResNet152的特征进行拼接
        fused_features = torch.cat((biformer_features, resnet_features), dim=1)
        
        # 通过MLP模型进行特征融合
        hidden_output = self.hidden_layer(fused_features)
        fused_output = self.output_layer(hidden_output)
        
        return fused_output

# 示例使用
biformer_dim = 512  # Biformer特征维度
resnet_dim = 2048  # ResNet152特征维度
hidden_dim = 256  # 隐藏层维度
output_dim = 10  # 输出维度，根据具体任务设定

# 创建模型实例
model = FeatureFusionMLP(biformer_dim, resnet_dim, hidden_dim, output_dim)

# 定义输入特征
biformer_features = torch.randn(1, biformer_dim)  # 假设输入为1个样本
resnet_features = torch.randn(1, resnet_dim)

# 进行特征融合
fused_output = model(biformer_features, resnet_features)

# 打印融合后的特征输出
print(fused_output)
