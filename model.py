import torch
import torch.nn as nn
import torch.nn.functional as F


class DrivingModel(nn.Module):
    """
    Multi-input CNN for end-to-end driving.

    Takes two inputs:
    - Main screen (224x224 RGB)
    - Minimap (128x128 RGB) with route line

    Outputs continuous values:
    - Steering: -1 (left) to 1 (right)
    - Throttle: 0 to 1
    - Brake: 0 to 1
    """

    def __init__(
        self, screen_input_shape=(3, 224, 224), minimap_input_shape=(3, 128, 128)
    ):
        super(DrivingModel, self).__init__()

        # Screen encoder (larger, more capacity)
        self.screen_conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Minimap encoder (smaller, focused on route detection)
        self.minimap_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        # Calculate flattened sizes
        screen_flat_size = self._get_conv_output_size(
            screen_input_shape, self.screen_conv
        )
        minimap_flat_size = self._get_conv_output_size(
            minimap_input_shape, self.minimap_conv
        )

        # Combined FC layers
        combined_size = screen_flat_size + minimap_flat_size

        self.fc = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

        # Output heads
        self.steering_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),  # Output: -1 to 1
        )

        self.throttle_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # Output: 0 to 1
        )

        self.brake_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # Output: 0 to 1
        )

    def _get_conv_output_size(self, input_shape, conv_layers):
        """Calculate the flattened size after conv layers"""
        dummy_input = torch.zeros(1, *input_shape)
        output = conv_layers(dummy_input)
        return int(torch.prod(torch.tensor(output.shape)))

    def forward(self, screen, minimap):
        # Encode screen
        screen_features = self.screen_conv(screen)
        screen_features = screen_features.view(screen_features.size(0), -1)

        # Encode minimap
        minimap_features = self.minimap_conv(minimap)
        minimap_features = minimap_features.view(minimap_features.size(0), -1)

        # Combine features
        combined = torch.cat([screen_features, minimap_features], dim=1)
        shared = self.fc(combined)

        # Get outputs
        steering = self.steering_head(shared)
        throttle = self.throttle_head(shared)
        brake = self.brake_head(shared)

        return steering, throttle, brake

    def predict(self, screen, minimap, device="cpu"):
        """Convenience method for inference"""
        self.eval()
        with torch.no_grad():
            screen = screen.to(device)
            minimap = minimap.to(device)
            steering, throttle, brake = self.forward(screen, minimap)
            return {
                "steering": steering.cpu().numpy().flatten(),
                "throttle": throttle.cpu().numpy().flatten(),
                "brake": brake.cpu().numpy().flatten(),
            }


class DrivingModelLight(nn.Module):
    """
    Lighter version for faster inference on CPU.
    Single input (screen only), smaller architecture.
    """

    def __init__(self, input_shape=(3, 224, 224)):
        super(DrivingModelLight, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        flat_size = self._get_conv_output_size(input_shape, self.features)

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        self.steering_head = nn.Linear(64, 1)
        self.throttle_head = nn.Linear(64, 1)
        self.brake_head = nn.Linear(64, 1)

    def _get_conv_output_size(self, input_shape, conv_layers):
        dummy_input = torch.zeros(1, *input_shape)
        output = conv_layers(dummy_input)
        return int(torch.prod(torch.tensor(output.shape)))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        steering = torch.tanh(self.steering_head(x))
        throttle = torch.sigmoid(self.throttle_head(x))
        brake = torch.sigmoid(self.brake_head(x))

        return steering, throttle, brake

    def predict(self, x, device="cpu"):
        self.eval()
        with torch.no_grad():
            x = x.to(device)
            steering, throttle, brake = self.forward(x)
            return {
                "steering": steering.cpu().numpy().flatten(),
                "throttle": throttle.cpu().numpy().flatten(),
                "brake": brake.cpu().numpy().flatten(),
            }


if __name__ == "__main__":
    # Test the models
    print("Testing DrivingModel...")
    model = DrivingModel()
    screen = torch.randn(2, 3, 224, 224)
    minimap = torch.randn(2, 3, 128, 128)
    steering, throttle, brake = model(screen, minimap)
    print(
        f"Steering shape: {steering.shape}, range: [{steering.min():.2f}, {steering.max():.2f}]"
    )
    print(
        f"Throttle shape: {throttle.shape}, range: [{throttle.min():.2f}, {throttle.max():.2f}]"
    )
    print(f"Brake shape: {brake.shape}, range: [{brake.min():.2f}, {brake.max():.2f}]")

    print("\nTesting DrivingModelLight...")
    model_light = DrivingModelLight()
    x = torch.randn(2, 3, 224, 224)
    steering, throttle, brake = model_light(x)
    print(
        f"Steering shape: {steering.shape}, range: [{steering.min():.2f}, {steering.max():.2f}]"
    )

    # Count parameters
    print(f"\nDrivingModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"DrivingModelLight parameters: {sum(p.numel() for p in model_light.parameters()):,}"
    )
