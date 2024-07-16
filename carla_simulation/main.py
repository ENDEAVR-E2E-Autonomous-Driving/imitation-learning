import pygame
import carla
import queue
import threading

from imitation_shared.input import InputManager
from imitation_shared.utils import *

from scene import CarlaScene, CarlaCamera
from data import DataManager
from model import load_model

from agents.navigation.local_planner import LocalPlanner, RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner

import logidrivepy

print_game_letterhead("Carla Imitation Learning")

game_width = 1920
game_height = 1080

# Initialize the Carla scene
scene = CarlaScene(town="Town02")
scene.open_window(w=game_width, h=game_height)

# Initialize the input and data managers
input_manager = InputManager()
save_queue = queue.Queue()

# Load the model (or an empty model if it doesn't exist)
model = load_model("data/model", "model_state_dict")

# Add a car to the scene
vehicle = scene.add_car()

# GRP and local planner for navigation ---------------------------------------------------------
import random
spawn_points = scene.world.get_map().get_spawn_points()

grp = GlobalRoutePlanner(scene.world.get_map(), sampling_resolution=4.0)
local_planner = LocalPlanner(vehicle.object, map_inst=scene.world.get_map())

start_waypoint = scene.world.get_map().get_waypoint(vehicle.get_spawn_point().location)
end_waypoint = scene.world.get_map().get_waypoint(random.choice(spawn_points).location)

route = grp.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)

local_planner.set_global_plan(route)

# ---------------------------------------------------------------------------------------------

# Add cameras to the scene
window_width, window_height = scene.get_window_size()
game_camera = CarlaCamera(vehicle.object, w=window_width, h=window_height, fov=110)
forward_camera = CarlaCamera(vehicle.object)
left_camera = CarlaCamera(vehicle.object, y=-0.25, rot=carla.Rotation(yaw=-5))
right_camera = CarlaCamera(vehicle.object, y=0.25, rot=carla.Rotation(yaw=5))

scene.add_game_camera(game_camera)
scene.add_camera(forward_camera)
scene.add_camera(left_camera)
scene.add_camera(right_camera)

running = True
collecting = False
autopilot = False
command = 1
distance_traveled = 0.0
command_cooldown_counter = 0
logitech_detected = True
navigate = False
change_weather = False

def save_data_thread():
    data_manager = DataManager("data/training", "training_data")

    while True:
        screenshot, scalars, targets, command = save_queue.get()
        if screenshot is None:
            break
        data_manager.save(screenshot, scalars, targets, command)


save_thread = threading.Thread(target=save_data_thread)
save_thread.start()

try:
    logitech = logidrivepy.LogitechController()
    logitech.steering_initialize()
except:
    logitech_detected = False

try:
    while running:
        scene.run()

        if navigate:
            local_planner.run_step()
            next_waypoint, waypoint_command = local_planner.get_incoming_waypoint_and_direction(steps=4)

            if waypoint_command == RoadOption.LEFT:
                command = 0
                command_cooldown_counter = 180
            elif waypoint_command == RoadOption.RIGHT:
                command = 2
                command_cooldown_counter = 180
            else:
                if command_cooldown_counter > 0:
                    command_cooldown_counter -= 1
                else:
                    command = 1

            if scene.frames % 30 == 0:
                start_waypoint = scene.world.get_map().get_waypoint(vehicle.object.get_location())
                route = grp.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)
                local_planner.set_global_plan(route)

            if local_planner.done() or len(local_planner._waypoints_queue) < 5:
                end_waypoint = scene.world.get_map().get_waypoint(random.choice(spawn_points).location)
                print("FINISHED ROUTE, GENERATING NEW ROUTE")

            for (wp_location, direction) in local_planner._waypoints_queue:
                wp_location = wp_location.transform.location
                wp_location.z += 15.0

                color = carla.Color(255, 0, 0) if direction == RoadOption.LEFT else carla.Color(0, 255, 0) if direction == RoadOption.RIGHT else carla.Color(0, 0, 255)

                scene.world.debug.draw_string(wp_location, "X", color=color, life_time=0.1)

        if change_weather:
            current_weather = scene.world.get_weather()

            weather = carla.WeatherParameters(
                sun_altitude_angle=(current_weather.sun_altitude_angle + 0.5) % 180)

            scene.world.set_weather(weather)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_c:
                    collecting = not collecting
                    print_formatted("Collecting data: %s" % collecting)
                elif event.key == pygame.K_p:
                    autopilot = not autopilot
                    print_formatted("Autopilot: %s" % autopilot)
                elif event.key == pygame.K_1:
                    command = 0
                    print_formatted("Command: 0")
                elif event.key == pygame.K_2:
                    command = 1
                    print_formatted("Command: 1")
                elif event.key == pygame.K_3:
                    command = 2
                    print_formatted("Command: 2")
                elif event.key == pygame.K_i:
                    input_manager.toggle_input_method()
                elif event.key == pygame.K_r:
                    autopilot = False
                    collecting = False
                    vehicle.reset()
                elif event.key == pygame.K_n:
                    navigate = not navigate
                    print_formatted("Navigation: %s" % navigate)
                elif event.key == pygame.K_m:
                    change_weather = not change_weather
                    print_formatted("Changing weather: %s" % change_weather)
            elif event.type == pygame.JOYBUTTONDOWN:
                if input_manager.get_button("collect"):
                    collecting = not collecting
                    print_formatted("Collecting data: %s" % collecting)
                elif input_manager.get_button("autopilot"):
                    autopilot = not autopilot
                    print_formatted("Autopilot: %s" % autopilot)
                elif input_manager.get_button("left"):
                    command = 0
                    print_formatted("Command: 0")
                elif input_manager.get_button("center"):
                    command = 1
                    print_formatted("Command: 1")
                elif input_manager.get_button("right"):
                    command = 2
                    print_formatted("Command: 2")

        steer, throttle, brake = input_manager.get_input()

        speed_limit = vehicle.object.get_speed_limit()
        gear = vehicle.object.get_control().gear
        scalars = [vehicle.get_velocity_norm(), speed_limit / 120.0, gear / 8.0]

        if collecting:
            if scene.frames % (30 / 15) == 0:
                left_image = left_camera.get_image_float()
                right_image = right_camera.get_image_float()
                forward_image = forward_camera.get_image_float()
                if forward_image is not None:
                    offset_factor = 0.10
                    steer_offset = max(min(offset_factor * (25.0 / max(vehicle.get_velocity(), 0.01)), 0.25), 0.01)

                    save_queue.put((forward_image, scalars, [steer, throttle, brake], command))
                    save_queue.put((left_image, scalars, [steer + steer_offset, throttle, brake], command))
                    save_queue.put((right_image, scalars, [steer - steer_offset, throttle, brake], command))
        elif autopilot:
            distance_traveled += vehicle.get_velocity() / 3600.0 / 30.0
            steer, throttle, brake = vehicle.get_autopilot_control(model, scalars, forward_camera.get_image_float(), command)

        if logitech_detected:
            if autopilot and not collecting:
                logitech.LogiPlaySpringForce(0, int(steer * 100), 50, 80)
                logitech.logi_update()
            else:
                logitech.LogiPlaySpringForce(0, 0, 30, 80)
                logitech.logi_update()

        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

        scene.render_steer(steer, x=50, y=75, scale=0.1)

        text_to_render = {
            "Speed": f"{vehicle.get_velocity():.1f}",
            "Speed Limit": f"{int(speed_limit)}",
            "Steer": f"{steer:.2f}",
            "Throttle": f"{throttle:.2f}",
            "Brake": f"{brake:.2f}",
            "Gear": f"{gear}",
            "Distance on Autopilot": f"{distance_traveled:.2f} km"
        }

        scene.render_text(text_to_render, x=0, y=game_height, anchor="bottomleft")

        text_to_render = {
            "Command": 'Left' if command == 0 else 'Center' if command == 1 else 'Right',
            "Collecting": str(collecting),
            "Autopilot": str(autopilot),
        }

        scene.render_text(text_to_render, x=game_width // 2, y=game_height - 100, anchor="midbottom", size=36)

        scene.update_display()

except KeyboardInterrupt:
    pass
finally:
    print_formatted("Exiting...", RED)
    save_queue.put((None, None, None, None))
    save_thread.join()
    print_formatted("Save thread joined, exiting...", RED)
    scene.cleanup()
    if logitech_detected:
        logitech.steering_shutdown()
