import os
import time
import torch
import numpy as np
import cv2
from collections import deque
from ImageGrab import grab_screen
import direct_input
import getkeys
from model_advanced import AdvancedDrivingModel, extract_route_geometry


class PIDController:
    """
    PID controller for smooth and precise steering control.
    Handles proportional, integral, and derivative components.
    """

    def __init__(
        self,
        kp=1.5,
        ki=0.05,
        kd=0.3,
        output_limit=1.0,
        integral_limit=0.5,
        sample_time=0.033,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.sample_time = sample_time

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0

        self.error_history = deque(maxlen=10)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0
        self.error_history.clear()

    def update(self, error, feed_forward=0.0):
        """
        Compute PID output.

        Args:
            error: Current error (target - actual)
            feed_forward: Optional feed-forward term for predictive control

        Returns:
            output: Control output in range [-output_limit, output_limit]
        """
        self.error_history.append(error)

        p_term = self.kp * error

        self.integral += error * self.sample_time
        self.integral = np.clip(
            self.integral, -self.integral_limit, self.integral_limit
        )
        i_term = self.ki * self.integral

        derivative = (
            (error - self.prev_error) / self.sample_time if self.prev_error != 0 else 0
        )
        d_term = self.kd * derivative

        output = p_term + i_term + d_term + feed_forward

        output = np.clip(output, -self.output_limit, self.output_limit)

        self.prev_error = error
        self.prev_output = output

        return output

    def set_gains(self, kp=None, ki=None, kd=None):
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd

    def adaptive_gains(self, speed):
        """
        Adjust PID gains based on vehicle speed.
        Lower gains at high speed for stability, higher at low speed for responsiveness.
        """
        speed_factor = np.clip(1.0 - (speed / 300.0), 0.5, 1.0)

        self.kp = 1.5 * speed_factor
        self.kd = 0.3 * speed_factor

    def get_debug_info(self):
        return {
            "p_term": self.kp * self.prev_error,
            "i_term": self.ki * self.integral,
            "d_term": self.kd
            * (
                (self.prev_error - getattr(self, "prev_prev_error", self.prev_error))
                / self.sample_time
            ),
            "integral": self.integral,
            "prev_error": self.prev_error,
        }


class TrajectorySmoother:
    """
    Uses predicted trajectory for smooth, anticipatory steering.
    Blends between immediate and predicted steering commands.
    """

    def __init__(self, prediction_horizon=10, blend_factor=0.3):
        self.prediction_horizon = prediction_horizon
        self.blend_factor = blend_factor
        self.trajectory_history = deque(maxlen=5)

    def smooth_steering(self, current_steering, trajectory):
        """
        Smooth steering using predicted trajectory.

        Args:
            current_steering: Current steering command
            trajectory: Predicted future steering angles (array of length prediction_horizon)

        Returns:
            smoothed_steering: Blended steering command
        """
        if trajectory is None or len(trajectory) < self.prediction_horizon:
            return current_steering

        self.trajectory_history.append(trajectory)

        if len(self.trajectory_history) >= 3:
            avg_trajectory = np.mean(list(self.trajectory_history), axis=0)
        else:
            avg_trajectory = trajectory

        lookahead_steering = avg_trajectory[min(3, len(avg_trajectory) - 1)]

        smoothed = (
            1 - self.blend_factor
        ) * current_steering + self.blend_factor * lookahead_steering

        return np.clip(smoothed, -1.0, 1.0)

    def reset(self):
        self.trajectory_history.clear()


class HybridController:
    """
    Hybrid controller combining neural network predictions with rule-based route following.
    Integrates:
    - Neural network steering prediction
    - Route detection from minimap
    - PID control for smooth output
    - Trajectory smoothing for anticipation
    """

    def __init__(
        self,
        pid_kp=1.5,
        pid_ki=0.05,
        pid_kd=0.3,
        route_weight=0.6,
        nn_weight=0.4,
        use_route_priority=True,
    ):
        self.pid = PIDController(kp=pid_kp, ki=pid_ki, kd=pid_kd)
        self.trajectory_smoother = TrajectorySmoother()

        self.route_weight = route_weight
        self.nn_weight = nn_weight
        self.use_route_priority = use_route_priority

        self.current_speed = 0.0
        self.last_route_direction = 0.0

    def compute_steering(
        self, nn_steering, route_direction, trajectory=None, route_confidence=1.0
    ):
        """
        Compute final steering command by blending all sources.

        Args:
            nn_steering: Steering from neural network
            route_direction: Steering from route detection
            trajectory: Predicted future steering angles
            route_confidence: Confidence in route detection (0-1)

        Returns:
            final_steering: Final steering command
        """
        effective_route_weight = self.route_weight * route_confidence
        effective_nn_weight = self.nn_weight

        total_weight = effective_route_weight + effective_nn_weight
        effective_route_weight /= total_weight
        effective_nn_weight /= total_weight

        blended_steering = (
            effective_route_weight * route_direction + effective_nn_weight * nn_steering
        )

        if trajectory is not None:
            blended_steering = self.trajectory_smoother.smooth_steering(
                blended_steering, trajectory
            )

        pid_output = self.pid.update(route_direction, feed_forward=nn_steering * 0.3)

        if self.use_route_priority and route_confidence > 0.8:
            final_steering = 0.7 * pid_output + 0.3 * blended_steering
        else:
            final_steering = blended_steering

        return np.clip(final_steering, -1.0, 1.0)

    def compute_throttle_brake(self, nn_throttle, nn_brake, steering_angle, speed=None):
        """
        Compute throttle and brake with speed-based adjustments.
        """
        steering_factor = 1.0 - 0.5 * abs(steering_angle)

        adjusted_throttle = nn_throttle * steering_factor

        if speed is not None and speed > 200:
            speed_factor = 1.0 - (speed - 200) / 200.0
            adjusted_throttle *= max(0.3, speed_factor)

            if abs(steering_angle) > 0.5 and nn_throttle > 0.5:
                adjusted_throttle *= 0.7
                adjusted_brake = max(nn_brake, 0.2)
            else:
                adjusted_brake = nn_brake
        else:
            adjusted_brake = nn_brake

        return adjusted_throttle, adjusted_brake

    def update_speed(self, speed):
        self.current_speed = speed
        self.pid.adaptive_gains(speed)

    def reset(self):
        self.pid.reset()
        self.trajectory_smoother.reset()
        self.current_speed = 0.0
        self.last_route_direction = 0.0


class FrameHistory:
    """
    Maintains history of frames for temporal modeling during inference.
    """

    def __init__(self, history_size=5, screen_size=(224, 224), minimap_size=(128, 128)):
        self.history_size = history_size
        self.screen_size = screen_size
        self.minimap_size = minimap_size

        self.screen_history = deque(maxlen=history_size - 1)

        self.dummy_screen = np.zeros((3, *screen_size), dtype=np.float32)

    def add_frame(self, screen, minimap):
        screen_preprocessed = self._preprocess_screen(screen)
        self.screen_history.append(screen_preprocessed)

    def get_screen_history(self):
        history = list(self.screen_history)
        while len(history) < self.history_size - 1:
            history.insert(0, self.dummy_screen.copy())
        return np.stack(history)

    def reset(self):
        self.screen_history.clear()

    def _preprocess_screen(self, screen):
        screen = cv2.resize(screen, self.screen_size)
        screen = screen.astype(np.float32) / 255.0
        screen = np.transpose(screen, (2, 0, 1))
        return screen


SCREEN_SIZE = 224
MINIMAP_SIZE = 128
GAME_REGION = (0, 40, 1024, 768)


def load_advanced_model(model_path, device="cpu"):
    """Load the advanced model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)

    frame_history = checkpoint.get("frame_history", 5)

    model = AdvancedDrivingModel(frame_history=frame_history, pretrained=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, frame_history


def find_latest_advanced_model(model_dir="models"):
    """Find the most recent advanced model file."""
    if not os.path.exists(model_dir):
        return None

    model_files = [
        f for f in os.listdir(model_dir) if f.endswith(".pt") and "advanced" in f
    ]
    if not model_files:
        model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]

    if not model_files:
        return None

    model_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(model_dir, f)), reverse=True
    )
    return os.path.join(model_dir, model_files[0])


def preprocess_screen(screen, target_size=(224, 224)):
    """Preprocess screen for model input."""
    screen = cv2.resize(screen, target_size)
    screen = screen.astype(np.float32) / 255.0
    screen = np.transpose(screen, (2, 0, 1))
    return torch.tensor(screen, dtype=torch.float32).unsqueeze(0)


def preprocess_minimap(minimap, target_size=(128, 128)):
    """Preprocess minimap for model input."""
    minimap = cv2.resize(minimap, target_size)
    minimap = minimap.astype(np.float32) / 255.0
    minimap = np.transpose(minimap, (2, 0, 1))
    return torch.tensor(minimap, dtype=torch.float32).unsqueeze(0)


def run_hybrid_self_driving(model_path=None, device=None, debug=True):
    """
    Run the hybrid self-driving system.
    Combines neural network predictions with route detection and PID control.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_path is None:
        model_path = find_latest_advanced_model()
        if model_path is None:
            print("No model found! Train a model first.")
            return
        print(f"Using model: {model_path}")

    model, frame_history = load_advanced_model(model_path, device)
    print(f"Loaded model with frame_history={frame_history}")

    controller = direct_input.XboxController()
    hybrid_controller = HybridController(
        pid_kp=1.5, pid_ki=0.05, pid_kd=0.3, route_weight=0.6, nn_weight=0.4
    )

    frame_buffer = FrameHistory(history_size=frame_history)

    minimap_x = 0
    minimap_y = GAME_REGION[3] - 250
    minimap_size = 250

    print("\n" + "=" * 60)
    print("HYBRID SELF-DRIVING MODE")
    print("=" * 60)
    print("Controls:")
    print("  Q - Quit")
    print("  P - Pause/Resume")
    print("  R - Reset PID controller")
    print("  Arrow keys - Adjust minimap position")
    print("  +/- - Adjust route weight")
    print("  [/] - Adjust NN weight")
    print("=" * 60)

    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

    print("Running!")

    paused = False
    fps_counter = 0
    fps = 0
    start_time = time.time()

    model.reset_hidden()

    try:
        while True:
            keys = getkeys.key_check()

            if "Q" in keys:
                print("Quitting...")
                break

            if "P" in keys:
                paused = not paused
                print("Paused" if paused else "Resumed")
                if paused:
                    controller.set_steering(0)
                    controller.set_throttle(0)
                    controller.set_brake(0)
                    hybrid_controller.reset()
                    frame_buffer.reset()
                    model.reset_hidden()
                time.sleep(0.3)

            if "R" in keys:
                hybrid_controller.reset()
                model.reset_hidden()
                print("Reset PID controller and model hidden state")
                time.sleep(0.3)

            if "LEFT" in keys:
                minimap_x = max(0, minimap_x - 10)
            if "RIGHT" in keys:
                minimap_x = min(GAME_REGION[2] - minimap_size, minimap_x + 10)
            if "UP" in keys:
                minimap_y = max(0, minimap_y - 10)
            if "DOWN" in keys:
                minimap_y = min(GAME_REGION[3] - minimap_size, minimap_y + 10)

            if "+" in keys or "=" in keys:
                hybrid_controller.route_weight = min(
                    0.9, hybrid_controller.route_weight + 0.05
                )
                print(f"Route weight: {hybrid_controller.route_weight:.2f}")
                time.sleep(0.1)
            if "-" in keys:
                hybrid_controller.route_weight = max(
                    0.1, hybrid_controller.route_weight - 0.05
                )
                print(f"Route weight: {hybrid_controller.route_weight:.2f}")
                time.sleep(0.1)

            if not paused:
                screen = grab_screen(region=GAME_REGION)

                minimap = screen[
                    minimap_y : minimap_y + minimap_size,
                    minimap_x : minimap_x + minimap_size,
                ]

                screen_tensor = preprocess_screen(
                    screen, (SCREEN_SIZE, SCREEN_SIZE)
                ).to(device)
                minimap_tensor = preprocess_minimap(
                    minimap, (MINIMAP_SIZE, MINIMAP_SIZE)
                ).to(device)

                screen_history = frame_buffer.get_screen_history()
                screen_history_tensor = (
                    torch.tensor(screen_history, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(device)
                )

                route_direction, route_points = extract_route_geometry(minimap)
                route_confidence = (
                    min(1.0, len(route_points) / 5.0) if route_points else 0.0
                )

                with torch.no_grad():
                    outputs = model(
                        screen_tensor,
                        minimap_tensor,
                        screen_history=screen_history_tensor,
                        return_trajectory=True,
                    )

                    nn_steering = outputs["steering"].item()
                    nn_throttle = outputs["throttle"].item()
                    nn_brake = outputs["brake"].item()
                    trajectory = outputs.get("trajectory")
                    if trajectory is not None:
                        trajectory = trajectory.cpu().numpy().flatten()

                final_steering = hybrid_controller.compute_steering(
                    nn_steering=nn_steering,
                    route_direction=route_direction,
                    trajectory=trajectory,
                    route_confidence=route_confidence,
                )

                throttle, brake = hybrid_controller.compute_throttle_brake(
                    nn_throttle=nn_throttle,
                    nn_brake=nn_brake,
                    steering_angle=final_steering,
                )

                controller.set_steering(final_steering)
                controller.set_throttle(throttle)
                controller.set_brake(brake)

                frame_buffer.add_frame(screen, minimap)

                fps_counter += 1
                if time.time() - start_time >= 1.0:
                    fps = fps_counter
                    fps_counter = 0
                    start_time = time.time()

                if debug:
                    display = cv2.resize(screen, (640, 480))

                    vis_minimap = minimap.copy()
                    if route_points:
                        for i, point in enumerate(route_points):
                            scale_x = vis_minimap.shape[1] / minimap_size
                            scale_y = vis_minimap.shape[0] / minimap_size
                            scaled_point = (
                                int(point[0] * scale_x),
                                int(point[1] * scale_y),
                            )
                            cv2.circle(vis_minimap, scaled_point, 3, (0, 255, 0), -1)
                            if i > 0:
                                prev_scaled = (
                                    int(route_points[i - 1][0] * scale_x),
                                    int(route_points[i - 1][1] * scale_y),
                                )
                                cv2.line(
                                    vis_minimap,
                                    prev_scaled,
                                    scaled_point,
                                    (0, 255, 0),
                                    1,
                                )

                    info_lines = [
                        f"FPS: {fps}",
                        f"Steering: {final_steering:.3f}",
                        f"NN: {nn_steering:.2f} | Route: {route_direction:.2f}",
                        f"Throttle: {throttle:.2f} | Brake: {brake:.2f}",
                        f"Route conf: {route_confidence:.2f}",
                        f"Route/NN weights: {hybrid_controller.route_weight:.1f}/{hybrid_controller.nn_weight:.1f}",
                    ]

                    for i, line in enumerate(info_lines):
                        cv2.putText(
                            display,
                            line,
                            (10, 25 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                    center_x = display.shape[1] // 2
                    center_y = display.shape[0] - 50
                    steering_x = int(center_x + final_steering * 100)
                    cv2.circle(display, (center_x, center_y), 30, (100, 100, 100), 2)
                    cv2.circle(display, (steering_x, center_y), 10, (0, 255, 255), -1)

                    if trajectory is not None:
                        for i, t in enumerate(trajectory[:5]):
                            future_x = int(center_x + t * 100)
                            future_y = center_y - (i + 1) * 15
                            cv2.circle(
                                display, (future_x, future_y), 3, (255, 100, 0), -1
                            )

                    cv2.imshow("Hybrid Self-Driving", display)
                    cv2.imshow("Minimap", cv2.resize(vis_minimap, (200, 200)))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        controller.set_steering(0)
        controller.set_throttle(0)
        controller.set_brake(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid self-driving system")
    parser.add_argument(
        "--model", type=str, default=None, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--no-debug", action="store_true", help="Disable debug visualization"
    )

    args = parser.parse_args()

    run_hybrid_self_driving(args.model, debug=not args.no_debug)
