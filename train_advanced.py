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
from collections import deque

from model_advanced import (
    AdvancedDrivingModel,
    AdvancedDrivingLoss,
    extract_route_geometry,
)


class AdvancedDrivingDataset(Dataset):
    """
    Advanced dataset with temporal frame stacking and route geometry extraction.
    """

    def __init__(
        self,
        data_path,
        frame_history=5,
        use_route_geometry=True,
        augment=True,
        screen_size=(224, 224),
        minimap_size=(128, 128),
    ):
        self.data_path = data_path
        self.frame_history = frame_history
        self.use_route_geometry = use_route_geometry
        self.augment = augment
        self.screen_size = screen_size
        self.minimap_size = minimap_size

        print(f"Loading data from {data_path}...")
        raw_data = np.load(data_path, allow_pickle=True)
        print(f"Loaded {len(raw_data)} samples")

        self.samples = self._process_raw_data(raw_data)
        print(f"Processed {len(self.samples)} valid samples")

        self._build_temporal_indices()

    def _process_raw_data(self, raw_data):
        samples = []

        for i, item in enumerate(raw_data):
            if len(item) >= 2:
                screen = item[0]
                controls = item[1]

                if isinstance(controls, (list, np.ndarray)):
                    if len(controls) == 9:
                        steering, throttle, brake = self._onehot_to_continuous(controls)
                    elif len(controls) == 3:
                        steering, throttle, brake = controls
                    else:
                        continue
                else:
                    continue

                minimap = item[2] if len(item) >= 3 else None

                samples.append(
                    {
                        "index": i,
                        "screen": screen,
                        "minimap": minimap,
                        "steering": float(steering),
                        "throttle": float(throttle),
                        "brake": float(brake),
                    }
                )

        return samples

    def _onehot_to_continuous(self, onehot):
        onehot = np.array(onehot)
        action = np.argmax(onehot)

        steering = 0.0
        throttle = 0.0
        brake = 0.0

        if action == 0:
            throttle = 1.0
        elif action == 1:
            brake = 1.0
        elif action == 2:
            steering = -1.0
        elif action == 3:
            steering = 1.0
        elif action == 4:
            throttle = 1.0
            steering = -0.7
        elif action == 5:
            throttle = 1.0
            steering = 0.7
        elif action == 6:
            brake = 1.0
            steering = -0.7
        elif action == 7:
            brake = 1.0
            steering = 0.7

        return steering, throttle, brake

    def _build_temporal_indices(self):
        """
        Build indices for temporal training.
        Each sample gets context from previous frames.
        """
        self.temporal_indices = []

        for i in range(len(self.samples)):
            history_indices = []
            for h in range(self.frame_history - 1, 0, -1):
                idx = max(0, i - h)
                history_indices.append(idx)
            history_indices.append(i)

            self.temporal_indices.append(
                {"history": history_indices[:-1], "current": i}
            )

    def _augment_image(self, img, is_minimap=False):
        if not self.augment:
            return img

        if random.random() < 0.5:
            brightness = random.uniform(0.7, 1.3)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)

        if random.random() < 0.3:
            contrast = random.uniform(0.8, 1.2)
            img = np.clip((img - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)

        if random.random() < 0.2:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        if not is_minimap and random.random() < 0.1:
            h, w = img.shape[:2]
            dx = random.randint(-10, 10)
            dy = random.randint(-5, 5)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        return img

    def _augment_steering(self, steering):
        if self.augment and random.random() < 0.3:
            noise = random.gauss(0, 0.03)
            steering = np.clip(steering + noise, -1.0, 1.0)
        return steering

    def _preprocess_screen(self, screen):
        if isinstance(screen, np.ndarray):
            screen = cv2.resize(screen, self.screen_size)
            screen = self._augment_image(screen)
            screen = screen.astype(np.float32) / 255.0
            screen = np.transpose(screen, (2, 0, 1))
        else:
            screen = np.zeros((3, *self.screen_size), dtype=np.float32)
        return screen

    def _preprocess_minimap(self, minimap):
        if isinstance(minimap, np.ndarray):
            minimap = cv2.resize(minimap, self.minimap_size)
            minimap = self._augment_image(minimap, is_minimap=True)
            minimap = minimap.astype(np.float32) / 255.0
            minimap = np.transpose(minimap, (2, 0, 1))
        else:
            minimap = np.zeros((3, *self.minimap_size), dtype=np.float32)
        return minimap

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        temporal_data = self.temporal_indices[idx]

        history_screens = []
        for h_idx in temporal_data["history"]:
            screen = self.samples[h_idx]["screen"]
            screen = self._preprocess_screen(screen)
            history_screens.append(screen)

        current_sample = self.samples[temporal_data["current"]]

        current_screen = self._preprocess_screen(current_sample["screen"])
        current_minimap = self._preprocess_minimap(current_sample["minimap"])

        steering = self._augment_steering(current_sample["steering"])

        route_direction = 0.0
        if self.use_route_geometry and current_sample["minimap"] is not None:
            route_direction, _ = extract_route_geometry(current_sample["minimap"])

        return {
            "screen": torch.tensor(current_screen, dtype=torch.float32),
            "screen_history": torch.tensor(
                np.stack(history_screens), dtype=torch.float32
            ),
            "minimap": torch.tensor(current_minimap, dtype=torch.float32),
            "steering": torch.tensor(steering, dtype=torch.float32),
            "throttle": torch.tensor(current_sample["throttle"], dtype=torch.float32),
            "brake": torch.tensor(current_sample["brake"], dtype=torch.float32),
            "route_direction_target": torch.tensor(
                route_direction, dtype=torch.float32
            ),
        }


def train_advanced_model(
    data_path="training_data_screen.npy",
    epochs=100,
    batch_size=16,
    lr=1e-4,
    frame_history=5,
    save_dir="models",
    device=None,
    use_pretrained=True,
    gradient_accumulation_steps=2,
):
    """
    Train the advanced driving model.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(save_dir, exist_ok=True)

    dataset = AdvancedDrivingDataset(
        data_path, frame_history=frame_history, use_route_geometry=True, augment=True
    )

    if len(dataset) == 0:
        print("No valid samples found!")
        return None

    val_split = 0.1
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    model = AdvancedDrivingModel(
        frame_history=frame_history, pretrained=use_pretrained
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    criterion = AdvancedDrivingLoss(
        steering_weight=3.0,
        throttle_weight=1.0,
        brake_weight=1.0,
        route_weight=1.5,
        trajectory_weight=2.0,
        smoothness_weight=0.5,
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=5
    )

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    prev_steering = None

    for epoch in range(epochs):
        if epoch < 5:
            warmup_scheduler.step()
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        model.train()
        train_loss = 0.0
        train_steering_loss = 0.0
        train_route_loss = 0.0

        optimizer.zero_grad()

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} [lr={current_lr:.2e}]"
        )

        for batch_idx, batch in enumerate(pbar):
            screen = batch["screen"].to(device)
            screen_history = batch["screen_history"].to(device)
            minimap = batch["minimap"].to(device)
            steering = batch["steering"].to(device)
            throttle = batch["throttle"].to(device)
            brake = batch["brake"].to(device)
            route_target = batch["route_direction_target"].to(device)

            outputs = model(
                screen, minimap, screen_history=screen_history, return_trajectory=True
            )

            targets = {
                "steering": steering,
                "throttle": throttle,
                "brake": brake,
                "route_direction_target": route_target,
            }

            losses = criterion(outputs, targets, prev_steering)

            loss = losses["total"] / gradient_accumulation_steps
            loss.backward()

            if prev_steering is None:
                prev_steering = steering.detach()
            else:
                prev_steering = 0.9 * prev_steering + 0.1 * steering.detach()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += losses["total"].item()
            train_steering_loss += losses["steering"].item()
            train_route_loss += losses["route"].item()

            pbar.set_postfix(
                {
                    "loss": f"{losses['total'].item():.4f}",
                    "steer": f"{losses['steering'].item():.4f}",
                    "route": f"{losses['route'].item():.4f}",
                }
            )

        train_loss /= len(train_loader)
        train_steering_loss /= len(train_loader)
        train_route_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        val_steering_loss = 0.0
        val_route_loss = 0.0
        model.reset_hidden()

        with torch.no_grad():
            for batch in val_loader:
                screen = batch["screen"].to(device)
                screen_history = batch["screen_history"].to(device)
                minimap = batch["minimap"].to(device)
                steering = batch["steering"].to(device)
                throttle = batch["throttle"].to(device)
                brake = batch["brake"].to(device)
                route_target = batch["route_direction_target"].to(device)

                outputs = model(
                    screen,
                    minimap,
                    screen_history=screen_history,
                    return_trajectory=True,
                )

                targets = {
                    "steering": steering,
                    "throttle": throttle,
                    "brake": brake,
                    "route_direction_target": route_target,
                }

                losses = criterion(outputs, targets)

                val_loss += losses["total"].item()
                val_steering_loss += losses["steering"].item()
                val_route_loss += losses["route"].item()

        val_loss /= len(val_loader)
        val_steering_loss /= len(val_loader)
        val_route_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(
            f"\nEpoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_steering={val_steering_loss:.4f}, val_route={val_route_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(save_dir, f"best_advanced_{int(time.time())}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "frame_history": frame_history,
                    "model_type": "advanced",
                },
                model_path,
            )
            print(f"  Saved best model to {model_path}")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                checkpoint_path,
            )
            print(f"  Saved checkpoint to {checkpoint_path}")

    final_path = os.path.join(save_dir, f"final_advanced_{int(time.time())}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "frame_history": frame_history,
            "model_type": "advanced",
        },
        final_path,
    )
    print(f"\nSaved final model to {final_path}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train advanced driving model")
    parser.add_argument(
        "--data",
        type=str,
        default="training_data_screen.npy",
        help="Path to training data",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--frame-history", type=int, default=5)
    parser.add_argument(
        "--no-pretrained", action="store_true", help="Disable pretrained backbone"
    )
    parser.add_argument("--save-dir", type=str, default="models")

    args = parser.parse_args()

    train_advanced_model(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        frame_history=args.frame_history,
        save_dir=args.save_dir,
        use_pretrained=not args.no_pretrained,
    )
