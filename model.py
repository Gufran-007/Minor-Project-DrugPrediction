import torch
import torch.nn as nn

class DrugNet(nn.Module):

    def __init__(self, input_dim):

        super(DrugNet, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 512),

            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),

            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),

            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.network(x)

if __name__ == '__main__':
    model = DrugNet(input_dim=5000)

    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal learnable parameters: {total_params:,}")

    fake_input = torch.randn(4, 5000)

    output = model(fake_input)
    print(f"\nTest input shape:  {fake_input.shape}")
    print(f"Test output shape: {output.shape}")
    print("\n✅ Model is working correctly!")