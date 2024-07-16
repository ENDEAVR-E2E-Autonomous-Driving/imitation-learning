from imitation_shared.utils import *
from data import *
import cv2
import numpy as np
import pygame

print_game_letterhead("HDF5 Data Viewer")

# Load the data
dataset = ImitationDataset("data/training")

if len(dataset) == 0:
    print_formatted("No data found in the training folder", RED)
    exit()

print_formatted(f"Loaded {len(dataset)} samples from the training folder", GREEN)

file_name = None

# Define text properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.3
font_color = (255, 255, 255)  # White color

video_width = 200 * 3
video_height = 88 + 70

fc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter("1.mp4", fc, 60, (video_width, video_height))

clock = pygame.time.Clock()

def render_image(data):
    image, scalars, targets, commands = data
    image = np.array(image, dtype=np.float32).transpose((1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = (image * 255).astype('uint8')

    targets_text = 'Targets: ' + ', '.join([f"{x:.4f}" for x in targets])
    cv2.putText(image, targets_text, (10, 10), font, font_scale, font_color)

    return image

# Loop through the data
for i in range(0, len(dataset), 3):
    # Render the images
    center_image = render_image(dataset[i])
    left_image = render_image(dataset[i + 1])
    right_image = render_image(dataset[i + 2])

    _, scalars, _, commands = dataset[i]

    file_name = dataset.get_file_for_index(i)

    file_name_text = 'File: ' + file_name
    scalars_text = 'Scalars: ' + ', '.join([f"{x:.2f}" for x in scalars])
    commands_text = 'Command: ' + str(commands)

    images = [left_image, center_image, right_image]
    concatenated_image = cv2.hconcat(images)
    # Add top border to the concatenated image
    concatenated_image = cv2.copyMakeBorder(concatenated_image, 70, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Overlay the text in the center of the concatenated image
    cv2.putText(concatenated_image, file_name_text, (10, 20), font, font_scale, font_color)
    cv2.putText(concatenated_image, scalars_text, (10, 35), font, font_scale, font_color)
    cv2.putText(concatenated_image, commands_text, (10, 50), font, font_scale, font_color)

    # Display the image with overlays
    cv2.imshow('Dataset Image', concatenated_image)
    cv2.imwrite("output.jpg", concatenated_image)
    video.write(concatenated_image)

    clock.tick(60)

    key = cv2.waitKey(1)
    # Break the loop if 'q' is pressed
    if key & 0xFF == ord('q'):
        break

print_formatted("End of dataset reached", RED)

cv2.destroyAllWindows()