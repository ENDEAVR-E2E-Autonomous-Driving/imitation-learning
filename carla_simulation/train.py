from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from model import *
from data import ImitationDataset
from preprocess import *

def main():
    print_game_letterhead("CARLA Simulation Training")

    print_formatted("Starting Training Process", GREEN)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_formatted(f"Using device: {GREEN}{device}{RESET}")

    # Load the model
    model = load_model("data/model", "model_state_dict")
    model = model.to(device)

    # Hyperparameters
    num_epochs = 30
    batch_size = 120

    # Tensorboard writer
    run_dir = f"runs/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    tsbd = SummaryWriter(log_dir=run_dir)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004, betas=(0.7, 0.85))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    try:
        optimizer.load_state_dict(torch.load("data/model/optimizer_state_dict.pth"))
    except FileNotFoundError:
        print_formatted("No optimizer state dictionary found", RED)

    # Load your entire dataset
    dataset = ImitationDataset("data/training")

    if len(dataset) == 0:
        print_formatted("No data found in the training folder", RED)
        return

    indices = get_balanced_commands("data/training")

    dataset = torch.utils.data.Subset(dataset, indices)

    # Determine the lengths of your splits
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation

    # Split the dataset
    train_dataset, validation_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders for the training and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    tsbd.add_graph(model, [torch.zeros(1, 3, 88, 200).to(device),
                           torch.zeros(1, 3).to(device),
                           torch.zeros(1, 1).to(device, dtype=torch.uint8)])

    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0

        # Training loop
        for i, (images_batch, scalars_batch, targets_batch, commands_batch) in enumerate(train_dataloader):
            images_batch = images_batch.to(device, non_blocking=True)
            scalars_batch = scalars_batch.to(device, non_blocking=True)
            targets_batch = targets_batch.to(device, non_blocking=True)
            commands_batch = commands_batch.to(device, non_blocking=True)

            outputs = model(images_batch, scalars_batch, commands_batch)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(outputs, targets_batch)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            running_loss += loss_val
            tsbd.add_scalar('Loss/Train', loss_val, epoch * len(train_dataloader) + i)

        # Validation loop
        model.eval()
        with torch.no_grad():
            validation_loss = 0.0
            for i, (images_batch, scalars_batch, targets_batch, commands_batch) in enumerate(validation_dataloader):
                images_batch = images_batch.to(device, non_blocking=True)
                scalars_batch = scalars_batch.to(device, non_blocking=True)
                targets_batch = targets_batch.to(device, non_blocking=True)
                commands_batch = commands_batch.to(device, non_blocking=True)

                outputs = model(images_batch, scalars_batch, commands_batch)
                loss = criterion(outputs, targets_batch)

                loss_val = loss.item()
                validation_loss += loss_val
                tsbd.add_scalar('Loss/Validation', loss_val, epoch * len(validation_dataloader) + i)

        # Save losses for plotting
        training_losses.append(running_loss / len(train_dataloader))
        validation_losses.append(validation_loss / len(validation_dataloader))

        tsbd.add_scalar('Loss/TrainEpoch', training_losses[-1], epoch)
        tsbd.add_scalar('Loss/ValidationEpoch', validation_losses[-1], epoch)
        tsbd.add_scalars("Loss/TrainValidation", {
            "Train": training_losses[-1],
            "Validation": validation_losses[-1]
        }, epoch)

        print_formatted(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {training_losses[-1]:.4f} - "
                        f"Validation Loss: {validation_losses[-1]:.4f}")

        model.train()
        lr_scheduler.step()

    # Save the model and plot the losses
    save_model("data/model", model, "model_state_dict")
    torch.save(optimizer.state_dict(), "data/model/optimizer_state_dict.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
