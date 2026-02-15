import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2


class RouteDetector(nn.Module):
    """
    Specialized module for detecting and encoding route geometry from minimap.
    Uses attention to focus on the blue route line.
    """

    def __init__(self, output_dim=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1), nn.Sigmoid())

        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )

        self.route_direction_head = nn.Linear(output_dim, 1)

    def forward(self, minimap):
        features = self.conv(minimap)

        attn_mask = self.attention(features)
        attended = features * attn_mask

        flat = attended.view(attended.size(0), -1)
        route_features = self.fc(flat)

        route_direction = torch.tanh(self.route_direction_head(route_features))

        return route_features, route_direction, attn_mask


class PretrainedScreenEncoder(nn.Module):
    """
    Screen encoder using pretrained ResNet18 backbone.
    Extracts rich visual features for driving decisions.
    """

    def __init__(self, output_dim=256, pretrained=True):
        super().__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        for i, layer in enumerate([resnet.layer1, resnet.layer2]):
            for param in layer.parameters():
                param.requires_grad = False

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TemporalLSTM(nn.Module):
    """
    LSTM module for temporal reasoning.
    Processes sequences of visual features to understand motion and predict trajectory.
    """

    def __init__(self, input_dim, hidden_dim=256, num_layers=2, output_dim=128):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=False,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out), hidden, lstm_out


class TrajectoryPredictor(nn.Module):
    """
    Predicts future steering trajectory for smoother control.
    Outputs sequence of future steering angles.
    """

    def __init__(self, input_dim, hidden_dim=64, horizon=10):
        super().__init__()

        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True
        )

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        dummy_input = x.unsqueeze(1).expand(-1, self.horizon, -1)

        out, _ = self.lstm(dummy_input)

        trajectory = torch.tanh(self.output_layer(out)).squeeze(-1)

        return trajectory


class AdvancedDrivingModel(nn.Module):
    """
    Advanced driving model combining:
    - Pretrained ResNet18 for screen encoding
    - Specialized route detector for minimap
    - LSTM for temporal reasoning
    - Trajectory prediction for smooth control
    - PID-like learned control refinement
    """

    def __init__(
        self,
        screen_input_shape=(3, 224, 224),
        minimap_input_shape=(3, 128, 128),
        frame_history=5,
        pretrained=True,
    ):
        super().__init__()

        self.frame_history = frame_history

        self.screen_encoder = PretrainedScreenEncoder(
            output_dim=256, pretrained=pretrained
        )

        self.route_detector = RouteDetector(output_dim=64)

        combined_dim = 256 + 64

        self.temporal_lstm = TemporalLSTM(
            input_dim=combined_dim, hidden_dim=256, num_layers=2, output_dim=128
        )

        self.trajectory_predictor = TrajectoryPredictor(
            input_dim=128, hidden_dim=64, horizon=10
        )

        self.steering_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

        self.throttle_head = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

        self.brake_head = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

        self.speed_head = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.ReLU()
        )

        self.hidden_state = None

    def reset_hidden(self):
        self.hidden_state = None

    def forward(self, screen, minimap, screen_history=None, return_trajectory=True):
        batch_size = screen.size(0)

        screen_features = self.screen_encoder(screen)

        route_features, route_direction, route_attention = self.route_detector(minimap)

        combined = torch.cat([screen_features, route_features], dim=1)

        if screen_history is not None:
            history_features = []
            for t in range(screen_history.size(1)):
                hist_screen = screen_history[:, t]
                hist_feat = self.screen_encoder(hist_screen)
                history_features.append(hist_feat)

            history_features = torch.stack(history_features, dim=1)
            history_route = torch.zeros(
                history_features.size(0),
                history_features.size(1),
                64,
                device=screen.device,
            )
            history_combined = torch.cat([history_features, history_route], dim=-1)

            current_combined = combined.unsqueeze(1)
            sequence = torch.cat([history_combined, current_combined], dim=1)
        else:
            sequence = combined.unsqueeze(1).expand(-1, self.frame_history, -1)

        temporal_features, self.hidden_state, lstm_sequence = self.temporal_lstm(
            sequence, self.hidden_state
        )

        steering = self.steering_head(temporal_features)
        throttle = self.throttle_head(temporal_features)
        brake = self.brake_head(temporal_features)
        speed_pred = self.speed_head(temporal_features)

        outputs = {
            "steering": steering,
            "throttle": throttle,
            "brake": brake,
            "speed_pred": speed_pred,
            "route_direction": route_direction,
            "route_attention": route_attention,
        }

        if return_trajectory:
            trajectory = self.trajectory_predictor(temporal_features)
            outputs["trajectory"] = trajectory

        return outputs

    def predict(self, screen, minimap, device="cpu", return_trajectory=True):
        self.eval()
        with torch.no_grad():
            screen = screen.to(device)
            minimap = minimap.to(device)

            outputs = self.forward(screen, minimap, return_trajectory=return_trajectory)

            return {
                "steering": outputs["steering"].cpu().numpy().flatten(),
                "throttle": outputs["throttle"].cpu().numpy().flatten(),
                "brake": outputs["brake"].cpu().numpy().flatten(),
                "route_direction": outputs["route_direction"].cpu().numpy().flatten(),
                "trajectory": outputs.get("trajectory", np.zeros(10))
                .cpu()
                .numpy()
                .flatten()
                if return_trajectory
                else None,
            }


class AdvancedDrivingLoss(nn.Module):
    """
    Multi-component loss function for advanced driving model.
    Combines:
    - Huber loss for steering (robust to outliers)
    - MSE for throttle/brake
    - Route adherence loss (align with detected route direction)
    - Trajectory consistency loss
    - Temporal smoothness loss
    """

    def __init__(
        self,
        steering_weight=3.0,
        throttle_weight=1.0,
        brake_weight=1.0,
        route_weight=1.5,
        trajectory_weight=2.0,
        smoothness_weight=0.5,
    ):
        super().__init__()

        self.steering_weight = steering_weight
        self.throttle_weight = throttle_weight
        self.brake_weight = brake_weight
        self.route_weight = route_weight
        self.trajectory_weight = trajectory_weight
        self.smoothness_weight = smoothness_weight

        self.huber = nn.HuberLoss(delta=0.1)
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, outputs, targets, prev_steering=None):
        steering_loss = self.huber(outputs["steering"].squeeze(), targets["steering"])

        throttle_loss = self.mse(outputs["throttle"].squeeze(), targets["throttle"])
        brake_loss = self.mse(outputs["brake"].squeeze(), targets["brake"])

        if "route_direction_target" in targets:
            route_loss = self.mse(
                outputs["route_direction"].squeeze(), targets["route_direction_target"]
            )
        else:
            route_loss = torch.tensor(0.0, device=outputs["steering"].device)

        if "trajectory" in outputs and "steering" in targets:
            current_steering = (
                targets["steering"]
                .unsqueeze(1)
                .expand(-1, outputs["trajectory"].size(1))
            )
            trajectory_loss = self.huber(outputs["trajectory"], current_steering)
        else:
            trajectory_loss = torch.tensor(0.0, device=outputs["steering"].device)

        smoothness_loss = torch.tensor(0.0, device=outputs["steering"].device)
        if prev_steering is not None:
            smoothness_loss = self.l1(outputs["steering"].squeeze(), prev_steering)

        total_loss = (
            self.steering_weight * steering_loss
            + self.throttle_weight * throttle_loss
            + self.brake_weight * brake_loss
            + self.route_weight * route_loss
            + self.trajectory_weight * trajectory_loss
            + self.smoothness_weight * smoothness_loss
        )

        return {
            "total": total_loss,
            "steering": steering_loss,
            "throttle": throttle_loss,
            "brake": brake_loss,
            "route": route_loss,
            "trajectory": trajectory_loss,
            "smoothness": smoothness_loss,
        }


def extract_route_geometry(minimap, device="cpu"):
    """
    Extract route direction from minimap using computer vision.
    Returns normalized steering direction based on route path.

    Args:
        minimap: numpy array (H, W, C) in BGR format
        device: torch device

    Returns:
        route_direction: float in range [-1, 1]
        route_points: list of (x, y) points along the route
    """
    if minimap is None or minimap.size == 0:
        return 0.0, []

    if isinstance(minimap, torch.Tensor):
        minimap = minimap.cpu().numpy()

    if minimap.ndim == 3 and minimap.shape[0] == 3:
        minimap = np.transpose(minimap, (1, 2, 0))

    if minimap.dtype == np.float32 or minimap.dtype == np.float64:
        minimap = (minimap * 255).astype(np.uint8)

    hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    points = np.column_stack(np.where(mask > 0))

    if len(points) == 0:
        return 0.0, []

    points = np.flip(points, axis=1)

    points = points[points[:, 1].argsort()]

    height, width = minimap.shape[:2]
    car_pos = (width // 2, height - 20)

    num_segments = 10
    segment_height = height // num_segments

    route_points = []
    for i in range(num_segments):
        y_min = height - (i + 1) * segment_height
        y_max = height - i * segment_height

        segment_points = points[(points[:, 1] >= y_min) & (points[:, 1] < y_max)]

        if len(segment_points) > 0:
            avg_x = int(np.mean(segment_points[:, 0]))
            avg_y = int(np.mean(segment_points[:, 1]))
            route_points.append((avg_x, avg_y))

    if len(route_points) < 2:
        return 0.0, route_points

    target_idx = min(3, len(route_points) - 1)
    target_point = route_points[target_idx]

    offset = target_point[0] - (width // 2)
    max_offset = width // 2.5
    route_direction = np.clip(offset / max_offset, -1.0, 1.0)

    return route_direction, route_points


if __name__ == "__main__":
    print("Testing AdvancedDrivingModel...")

    model = AdvancedDrivingModel(pretrained=False)

    screen = torch.randn(2, 3, 224, 224)
    minimap = torch.randn(2, 3, 128, 128)

    outputs = model(screen, minimap)

    print(
        f"Steering: {outputs['steering'].shape}, range: [{outputs['steering'].min():.2f}, {outputs['steering'].max():.2f}]"
    )
    print(f"Throttle: {outputs['throttle'].shape}")
    print(f"Brake: {outputs['brake'].shape}")
    print(f"Trajectory: {outputs['trajectory'].shape}")
    print(f"Route direction: {outputs['route_direction'].shape}")
    print(f"Route attention: {outputs['route_attention'].shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    loss_fn = AdvancedDrivingLoss()
    targets = {
        "steering": torch.tensor([0.5, -0.3]),
        "throttle": torch.tensor([0.8, 0.6]),
        "brake": torch.tensor([0.0, 0.1]),
    }
    losses = loss_fn(outputs, targets)
    print(f"\nLoss breakdown:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
