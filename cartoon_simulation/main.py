import pygame
import threading
import queue

from imitation_shared.utils import *
from imitation_shared.input import InputManager
from scene import Scene, Car
from data import DataManager
from model import load_model

"""
2D Imitation Learning Application
CSCE 482 - Capstone
Developed by Preston Barnett, Glenn Edstrom, Kirk Graham, and Isabelle Grimesey

This is the main entry point for the 2D Imitation Learning application. 
It initializes the game scene and input.

--------------------- CONTROLS ---------------------
Car:
    - Keyboard Control:
        - L/R Arrow Keys: Steer the car
        - Up/Down Arrow Keys: Accelerate/Decelerate
        
    - Joystick Control:
        - Mapped to the first joystick detected

Program:
    - i: Toggle input method (keyboard/joystick)
    - q: Quit the application
----------------------------------------------------
"""

print_game_letterhead()

args = parse_args()
sampling_rate = args.sampling_rate
config_path = args.config

print_args(args)

# Load scene prerequisites
background_image = pygame.image.load('./scene_assets/racetrack.jpeg')
car_image = pygame.image.load('./scene_assets/car.png')
initial_x = 800
initial_y = 600
initial_angle = 0

# Initialize the game scene and input manager
scene = Scene(background_image, initial_x, initial_y, initial_angle)
input_manager = InputManager()

# Initialize the data manager
data_manager = DataManager("data/training", "training_data")
save_queue = queue.Queue()

# Load the model (or use an unweighted model if none is found)
model = load_model("data/model", "model_state_dict")

# Create the agents and add them to the scene
car_agent = Car(300, 300, 0, 0.1)
scene.add_agent(car_agent)

# Main game loop
frames = 0
running = True
collecting = False
autopilot = False

def save_data_thread():
    while True:
        screenshot, velocity, controls = save_queue.get()
        if screenshot is None:
            break
        data_manager.save(screenshot, [velocity], controls)

save_thread = threading.Thread(target=save_data_thread)
save_thread.start()

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_i:
                    input_manager.toggle_input_method()
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_c:
                    collecting = not collecting
                    if collecting:
                        print_formatted("Collecting data...", GREEN)
                    else:
                        print_formatted("Stopped collecting data...", RED)
                if event.key == pygame.K_a:
                    autopilot = not autopilot
                    if autopilot:
                        print_formatted("Autopilot enabled...", GREEN)
                    else:
                        print_formatted("Autopilot disabled...", RED)

        steer, throttle, brake = input_manager.get_input()
        scene.run()

        frames += 1

        if collecting and frames % (60 / sampling_rate) == 0:
            save_queue.put((scene.take_screenshot(car_agent), car_agent.velocity, [steer, throttle, brake]))
        if autopilot:
            steer, throttle, brake = car_agent.get_autopilot_control(scene, model)

        car_agent.update_physics(steer, throttle, brake)
except KeyboardInterrupt:
    print_formatted("KeyboardInterrupt detected, exiting...", RED)
    running = False
finally:
    print_formatted("Exiting...", RED)
    save_queue.put((None, None, None))
    save_thread.join()
    print_formatted("Save thread joined, exiting...", RED)
    pygame.quit()
