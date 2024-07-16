import numpy as np
from data import ImitationDataset


def get_balanced_commands(dataset_name):
    dataset = ImitationDataset(dataset_name, include_image=False)

    left_indices = []
    center_indices = []
    right_indices = []

    for i in range(len(dataset)):
        _, _, _, command = dataset[i]
        command = command.item()
        if command == 0:
            left_indices.append(i)
        elif command == 1:
            center_indices.append(i)
        elif command == 2:
            right_indices.append(i)

    minimum = min(len(left_indices), len(center_indices), len(right_indices))

    np.random.shuffle(left_indices)
    np.random.shuffle(center_indices)
    np.random.shuffle(right_indices)

    left_indices = left_indices[:minimum]
    center_indices = center_indices[:minimum]
    right_indices = right_indices[:minimum]

    indices = left_indices + center_indices + right_indices

    return indices


if __name__ == "__main__":
    indices = get_balanced_commands("data/training")
    print(len(indices))
