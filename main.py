import os
import sys
import glob


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def main():
    while True:
        clear_screen()
        print("=" * 60)
        print("SelfDriving4Forza - Advanced PyTorch Edition")
        print("=" * 60)
        print("")
        print("DATA COLLECTION")
        print("  1. Collect Training Data")
        print("  2. Merge Training Data Files")
        print("")
        print("TRAINING")
        print("  3. Train Basic Model (screen-only)")
        print("  4. Train Advanced Model (hybrid + temporal)")
        print("")
        print("TESTING")
        print("  5. Test Basic Model (self-driving mode)")
        print("  6. Test Advanced Model (hybrid control)")
        print("")
        print("UTILITIES")
        print("  7. Evaluate Model")
        print("  8. Show Training Data Statistics")
        print("  9. Quit")
        print("=" * 60)
        print("\nRecommended Workflows:")
        print("  Basic:    1 -> 3 -> 5  (collect, train, test)")
        print("  Advanced: 1 -> 4 -> 6  (collect, train advanced, test hybrid)")
        print("=" * 60)

        choice = input("\nEnter your choice (1-9): ")

        if choice == "1":
            from collect_data import collect_training_data

            collect_training_data()
            input("\nPress Enter to return to menu...")

        elif choice == "2":
            from collect_data import merge_data_files

            print("\nMerge Training Data Files")
            print("-" * 30)
            pattern = (
                input("File pattern (default: training_data_*.npy): ")
                or "training_data_*.npy"
            )
            output = (
                input("Output filename (default: merged_training_data.npy): ")
                or "merged_training_data.npy"
            )
            merge_data_files(pattern, output)
            input("\nPress Enter to return to menu...")

        elif choice == "3":
            from train_pytorch import train_model

            print("\nTrain Basic Model")
            print("-" * 30)

            data_files = glob.glob("training_data_*.npy") + glob.glob(
                "merged_training_data.npy"
            )
            if data_files:
                print("Available data files:")
                for i, f in enumerate(data_files):
                    print(f"  {i + 1}. {f}")
                file_choice = input(
                    f"Select file (1-{len(data_files)}) or enter path: "
                )
                try:
                    idx = int(file_choice) - 1
                    data_path = data_files[idx]
                except (ValueError, IndexError):
                    data_path = (
                        file_choice if file_choice else "merged_training_data.npy"
                    )
            else:
                data_path = (
                    input("Enter data file path: ") or "training_data_screen.npy"
                )

            use_minimap = input("Use minimap input? (y/n, default: n): ").lower() == "y"
            epochs = input("Epochs (default: 50): ")
            epochs = int(epochs) if epochs else 50
            batch_size = input("Batch size (default: 32): ")
            batch_size = int(batch_size) if batch_size else 32
            lr = input("Learning rate (default: 0.0001): ")
            lr = float(lr) if lr else 1e-4

            train_model(
                data_path=data_path,
                use_minimap=use_minimap,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
            )
            input("\nPress Enter to return to menu...")

        elif choice == "4":
            from train_advanced import train_advanced_model

            print("\nTrain Advanced Model")
            print("-" * 30)
            print(
                "Features: Pretrained ResNet18 + LSTM + Route Detection + PID Control"
            )
            print("-" * 30)

            data_files = glob.glob("training_data_*.npy") + glob.glob(
                "merged_training_data.npy"
            )
            if data_files:
                print("Available data files:")
                for i, f in enumerate(data_files):
                    print(f"  {i + 1}. {f}")
                file_choice = input(
                    f"Select file (1-{len(data_files)}) or enter path: "
                )
                try:
                    idx = int(file_choice) - 1
                    data_path = data_files[idx]
                except (ValueError, IndexError):
                    data_path = (
                        file_choice if file_choice else "merged_training_data.npy"
                    )
            else:
                data_path = (
                    input("Enter data file path: ") or "training_data_screen.npy"
                )

            use_pretrained = (
                input("Use pretrained backbone? (y/n, default: y): ").lower() != "n"
            )
            epochs = input("Epochs (default: 100): ")
            epochs = int(epochs) if epochs else 100
            batch_size = input("Batch size (default: 16): ")
            batch_size = int(batch_size) if batch_size else 16
            lr = input("Learning rate (default: 0.0001): ")
            lr = float(lr) if lr else 1e-4
            frame_history = input("Frame history (default: 5): ")
            frame_history = int(frame_history) if frame_history else 5

            train_advanced_model(
                data_path=data_path,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                frame_history=frame_history,
                use_pretrained=use_pretrained,
            )
            input("\nPress Enter to return to menu...")

        elif choice == "5":
            from test_pytorch import run_self_driving

            print("\nBasic Self-Driving Mode")
            print("-" * 30)

            model_files = glob.glob("models/*.pt")
            advanced_files = [f for f in model_files if "advanced" in f]
            basic_files = [f for f in model_files if "advanced" not in f]

            if basic_files:
                print("Available basic models:")
                for i, f in enumerate(basic_files):
                    print(f"  {i + 1}. {f}")
                file_choice = input(
                    f"Select model (1-{len(basic_files)}) or press Enter for latest: "
                )
                try:
                    idx = int(file_choice) - 1
                    model_path = basic_files[idx]
                except (ValueError, IndexError):
                    model_path = None
            else:
                model_path = None

            use_minimap = (
                input("Use minimap? (y/n, default: auto-detect): ").lower() == "y"
            )

            run_self_driving(model_path=model_path, use_minimap=use_minimap)
            input("\nPress Enter to return to menu...")

        elif choice == "6":
            from test_advanced import run_hybrid_self_driving

            print("\nAdvanced Hybrid Self-Driving Mode")
            print("-" * 30)
            print("Features: PID control + Route following + Trajectory prediction")
            print("-" * 30)

            model_files = glob.glob("models/*advanced*.pt")
            if not model_files:
                model_files = glob.glob("models/*.pt")

            if model_files:
                print("Available models:")
                for i, f in enumerate(model_files):
                    print(f"  {i + 1}. {f}")
                file_choice = input(
                    f"Select model (1-{len(model_files)}) or press Enter for latest: "
                )
                try:
                    idx = int(file_choice) - 1
                    model_path = model_files[idx]
                except (ValueError, IndexError):
                    model_path = None
            else:
                model_path = None

            run_hybrid_self_driving(model_path=model_path)
            input("\nPress Enter to return to menu...")

        elif choice == "7":
            from test_pytorch import evaluate_model

            evaluate_model()
            input("\nPress Enter to return to menu...")

        elif choice == "8":
            import numpy as np

            print("\nTraining Data Statistics")
            print("-" * 30)

            data_files = glob.glob("training_data_*.npy") + glob.glob(
                "merged_training_data.npy"
            )

            for data_file in data_files:
                print(f"\n{data_file}:")
                try:
                    data = np.load(data_file, allow_pickle=True)
                    print(f"  Total samples: {len(data)}")

                    steerings = []
                    throttles = []
                    brakes = []

                    for sample in data:
                        if len(sample) >= 2:
                            controls = sample[1]
                            if isinstance(controls, (list, np.ndarray)):
                                if len(controls) == 3:
                                    steerings.append(controls[0])
                                    throttles.append(controls[1])
                                    brakes.append(controls[2])
                                elif len(controls) == 9:
                                    action = np.argmax(controls)
                                    if action == 2:
                                        steerings.append(-1.0)
                                    elif action == 3:
                                        steerings.append(1.0)
                                    elif action in [0, 4, 5]:
                                        steerings.append(
                                            -0.7
                                            if action == 4
                                            else (0.7 if action == 5 else 0.0)
                                        )
                                        throttles.append(1.0)
                                    elif action in [1, 6, 7]:
                                        brakes.append(1.0)

                    if steerings:
                        print(
                            f"  Steering: mean={np.mean(steerings):.3f}, std={np.std(steerings):.3f}"
                        )
                        print(
                            f"  Throttle active: {sum(1 for t in throttles if t > 0.5)} samples"
                        )
                        print(
                            f"  Brake active: {sum(1 for b in brakes if b > 0.5)} samples"
                        )

                        left = sum(1 for s in steerings if s < -0.5)
                        straight = sum(1 for s in steerings if -0.5 <= s <= 0.5)
                        right = sum(1 for s in steerings if s > 0.5)

                        print(
                            f"  Distribution: L={left} ({100 * left / len(steerings):.1f}%), "
                            f"S={straight} ({100 * straight / len(steerings):.1f}%), "
                            f"R={right} ({100 * right / len(steerings):.1f}%)"
                        )
                except Exception as e:
                    print(f"  Error loading: {e}")

            input("\nPress Enter to return to menu...")

        elif choice == "9":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
