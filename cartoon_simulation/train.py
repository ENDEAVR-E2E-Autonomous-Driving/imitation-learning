from torch.utils.data import DataLoader
from torch.utils.data import random_split

import matplotlib.pyplot as plt

from model import *
from data import ImitationDataset


def main():
    print_game_letterhead("2D Simulation Training")

    print_formatted("Starting Training Process", GREEN)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_formatted(f"Using device: {GREEN}{device}{RESET}")

    # Load the model
    model = load_model("data/model", "model_state_dict")
    model = model.to(device)

    # Hyperparameters
    num_epochs = 15
    batch_size = 128

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Load your entire dataset
    dataset = ImitationDataset("data/training")

    if len(dataset) == 0:
        print_formatted("No data found in the training folder", RED)
        return

    # Determine the lengths of your splits
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation

    # Split the dataset
    train_dataset, validation_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders for the training and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0

        # Training loop
        for images_batch, scalars_batch, targets_batch in train_dataloader:
            images_batch = images_batch.to(device, non_blocking=True)
            scalars_batch = scalars_batch.to(device, non_blocking=True)
            targets_batch = targets_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images_batch, scalars_batch)
            loss = criterion(outputs, targets_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation loop
        model.eval()
        with torch.no_grad():
            validation_loss = 0.0
            for images_batch, scalars_batch, targets_batch in validation_dataloader:
                images_batch, scalars_batch, targets_batch = images_batch.to(device), scalars_batch.to(
                    device), targets_batch.to(device)

                outputs = model(images_batch, scalars_batch)
                loss = criterion(outputs, targets_batch)

                validation_loss += loss.item()

        # Save losses for plotting
        training_losses.append(running_loss / len(train_dataloader))
        validation_losses.append(validation_loss / len(validation_dataloader))

        print_formatted(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {training_losses[-1]:.4f} - "
                        f"Validation Loss: {validation_losses[-1]:.4f}")

        model.train()

    # Save the model and plot the losses
    save_model("data/model", model, "model_state_dict")

    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
