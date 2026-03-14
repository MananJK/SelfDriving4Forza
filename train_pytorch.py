import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import cv2
import random
from tqdm import tqdm
import time

from model import DrivingModel, DrivingModelLight


class DrivingDataset(Dataset):
    """
    Dataset for driving data with augmentation support.

    Expected data format:
    - training_data.npy: array of [screen_img, minimap_img, [steering, throttle, brake]]

    If minimap not available, use screen-only mode.
    """

    def __init__(
        self,
        data_path,
        use_minimap=True,
        augment=True,
        screen_size=(224, 224),
        minimap_size=(128, 128),
    ):
        self.data_path = data_path
        self.use_minimap = use_minimap
        self.augment = augment
        self.screen_size = screen_size
        self.minimap_size = minimap_size

        print(f"Loading data from {data_path}...")
        raw_data = np.load(data_path, allow_pickle=True)
        print(f"Loaded {len(raw_data)} samples")

        self.samples = self._process_raw_data(raw_data)
        print(f"Processed {len(self.samples)} valid samples")

    def _process_raw_data(self, raw_data):
        """Convert raw data to usable format"""
        samples = []

        for item in raw_data:
            if len(item) >= 2:
                screen = item[0]
                controls = item[1]

                # Handle different control formats
                if isinstance(controls, (list, np.ndarray)):
                    if len(controls) == 9:
                        # Old one-hot format: convert to continuous
                        steering, throttle, brake = self._onehot_to_continuous(controls)
                    elif len(controls) == 3:
                        steering, throttle, brake = controls
                    else:
                        continue
                else:
                    continue

                # Check for minimap
                minimap = item[2] if len(item) >= 3 else None

                samples.append(
                    {
                        "screen": screen,
                        "minimap": minimap,
                        "steering": float(steering),
                        "throttle": float(throttle),
                        "brake": float(brake),
                    }
                )

        return samples

    def _onehot_to_continuous(self, onehot):
        """Convert old 9-class one-hot to continuous steering/throttle/brake"""
        onehot = np.array(onehot)
        action = np.argmax(onehot)

        # [W, S, A, D, WA, WD, SA, SD, NO_INPUT]
        steering = 0.0
        throttle = 0.0
        brake = 0.0

        if action == 0:  # W - Forward
            throttle = 1.0
        elif action == 1:  # S - Backward
            brake = 1.0
        elif action == 2:  # A - Left
            steering = -1.0
        elif action == 3:  # D - Right
            steering = 1.0
        elif action == 4:  # WA - Forward+Left
            throttle = 1.0
            steering = -0.7
        elif action == 5:  # WD - Forward+Right
            throttle = 1.0
            steering = 0.7
        elif action == 6:  # SA - Backward+Left
            brake = 1.0
            steering = -0.7
        elif action == 7:  # SD - Backward+Right
            brake = 1.0
            steering = 0.7
        # action == 8: No input - all zeros

        return steering, throttle, brake

    def _augment_image(self, img, is_minimap=False):
        """Apply data augmentation to image"""
        if not self.augment:
            return img

        # Random brightness adjustment
        if random.random() < 0.5:
            brightness = random.uniform(0.7, 1.3)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)

        # Random contrast adjustment
        if random.random() < 0.3:
            contrast = random.uniform(0.8, 1.2)
            img = np.clip((img - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)

        # Random blur (simulate motion blur)
        if random.random() < 0.2:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        return img

    def _augment_steering(self, steering):
        """Add small random noise to steering during training"""
        if self.augment and random.random() < 0.3:
            noise = random.gauss(0, 0.05)
            steering = np.clip(steering + noise, -1.0, 1.0)
        return steering

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Process screen
        screen = sample["screen"]
        if isinstance(screen, np.ndarray):
            screen = cv2.resize(screen, self.screen_size)
            screen = self._augment_image(screen)
            screen = screen.astype(np.float32) / 255.0
            screen = np.transpose(screen, (2, 0, 1))  # HWC to CHW
        else:
            screen = np.zeros((3, *self.screen_size), dtype=np.float32)

        # Process minimap if available
        if self.use_minimap and sample["minimap"] is not None:
            minimap = sample["minimap"]
            if isinstance(minimap, np.ndarray):
                minimap = cv2.resize(minimap, self.minimap_size)
                minimap = self._augment_image(minimap, is_minimap=True)
                minimap = minimap.astype(np.float32) / 255.0
                minimap = np.transpose(minimap, (2, 0, 1))
            else:
                minimap = np.zeros((3, *self.minimap_size), dtype=np.float32)
        else:
            minimap = np.zeros((3, *self.minimap_size), dtype=np.float32)

        # Get controls with augmentation
        steering = self._augment_steering(sample["steering"])

        return {
            "screen": torch.tensor(screen, dtype=torch.float32),
            "minimap": torch.tensor(minimap, dtype=torch.float32),
            "steering": torch.tensor(steering, dtype=torch.float32),
            "throttle": torch.tensor(sample["throttle"], dtype=torch.float32),
            "brake": torch.tensor(sample["brake"], dtype=torch.float32),
        }


class SteeringLoss(nn.Module):
    """
    Custom loss function that emphasizes steering accuracy.
    Uses weighted combination of MSE for each output.
    """

    def __init__(self, steering_weight=2.0, throttle_weight=1.0, brake_weight=1.0):
        super().__init__()
        self.steering_weight = steering_weight
        self.throttle_weight = throttle_weight
        self.brake_weight = brake_weight
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_steering,
        pred_throttle,
        pred_brake,
        target_steering,
        target_throttle,
        target_brake,
    ):
        steering_loss = self.mse(pred_steering.squeeze(), target_steering)
        throttle_loss = self.mse(pred_throttle.squeeze(), target_throttle)
        brake_loss = self.mse(pred_brake.squeeze(), target_brake)

        total_loss = (
            self.steering_weight * steering_loss
            + self.throttle_weight * throttle_loss
            + self.brake_weight * brake_loss
        )

        return total_loss, steering_loss, throttle_loss, brake_loss


def train_model(
    data_path="training_data_screen.npy",
    use_minimap=False,
    epochs=50,
    batch_size=32,
    lr=1e-4,
    save_dir="models",
    device=None,
):
    """
    Train the driving model.

    Args:
        data_path: Path to training data .npy file
        use_minimap: Whether to use minimap input (requires data with minimap)
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        save_dir: Directory to save model checkpoints
        device: torch device (auto-detected if None)
    """
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    full_dataset = DrivingDataset(data_path, use_minimap=use_minimap, augment=True)

    if len(full_dataset) == 0:
        print("No valid samples found!")
        return None

    # Split into train/val
    val_split = 0.1
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Create model
    if use_minimap:
        model = DrivingModel().to(device)
        print("Using DrivingModel (screen + minimap)")
    else:
        model = DrivingModelLight().to(device)
        print("Using DrivingModelLight (screen only)")

    # Loss and optimizer
    criterion = SteeringLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_steering_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            screen = batch["screen"].to(device)
            steering = batch["steering"].to(device)
            throttle = batch["throttle"].to(device)
            brake = batch["brake"].to(device)

            optimizer.zero_grad()

            if use_minimap:
                minimap = batch["minimap"].to(device)
                pred_steering, pred_throttle, pred_brake = model(screen, minimap)
            else:
                pred_steering, pred_throttle, pred_brake = model(screen)

            loss, steer_loss, throttle_loss, brake_loss = criterion(
                pred_steering, pred_throttle, pred_brake, steering, throttle, brake
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_steering_loss += steer_loss.item()

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "steer": f"{steer_loss.item():.4f}"}
            )

        train_loss /= len(train_loader)
        train_steering_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steering_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                screen = batch["screen"].to(device)
                steering = batch["steering"].to(device)
                throttle = batch["throttle"].to(device)
                brake = batch["brake"].to(device)

                if use_minimap:
                    minimap = batch["minimap"].to(device)
                    pred_steering, pred_throttle, pred_brake = model(screen, minimap)
                else:
                    pred_steering, pred_throttle, pred_brake = model(screen)

                loss, steer_loss, _, _ = criterion(
                    pred_steering, pred_throttle, pred_brake, steering, throttle, brake
                )

                val_loss += loss.item()
                val_steering_loss += steer_loss.item()

        val_loss /= len(val_loader)
        val_steering_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_steering={val_steering_loss:.4f}"
        )

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(save_dir, f"best_model_{int(time.time())}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "use_minimap": use_minimap,
                },
                model_path,
            )
            print(f"  Saved best model to {model_path}")

    # Save final model
    final_path = os.path.join(save_dir, f"final_model_{int(time.time())}.pt")
    torch.save(
        {"model_state_dict": model.state_dict(), "use_minimap": use_minimap}, final_path
    )
    print(f"Saved final model to {final_path}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train driving model")
    parser.add_argument(
        "--data",
        type=str,
        default="training_data_screen.npy",
        help="Path to training data",
    )
    parser.add_argument("--minimap", action="store_true", help="Use minimap input")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    train_model(
        data_path=args.data,
        use_minimap=args.minimap,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
