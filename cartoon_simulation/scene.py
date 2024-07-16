import pygame
import math
import torch


class Scene:
    """
    Represents a scene in a 2D simulation with background, agents, and handling of basic
    scene updates and rendering.

    Attributes:
        background_image (pygame.Surface): The background image of the scene.
        x (int): Initial x position of the scene.
        y (int): Initial y position of the scene.
        angle (float): Initial angle of the scene.
        screen_width (int): Width of the screen derived from the background image.
        screen_height (int): Height of the screen derived from the background image.
        clock (pygame.time.Clock): Clock used to manage update rates.
        agents (list): A list of agents (e.g., cars) added to the scene.
        screen (pygame.Surface): The pygame display surface.
    """

    def __init__(self, background_image, initial_x, initial_y, initial_angle):
        """
        Initializes the scene with a background, position, and angle. Sets up the
        screen and joystick if available.

        Parameters:
            config (dict): Configuration settings for the scene.
            background_image (pygame.Surface): Background image for the scene.
            initial_x (int): Initial x position for the scene.
            initial_y (int): Initial y position for the scene.
            initial_angle (float): Initial angle for the scene.
        """
        self.background_image = background_image
        self.x = initial_x
        self.y = initial_y
        self.angle = initial_angle
        self.screen_width = background_image.get_width()
        self.screen_height = background_image.get_height()
        self.clock = None
        self.agents = []

        self.screen = self.initialize_screen()

    def initialize_screen(self):
        """
        Initializes the pygame display surface based on the background image dimensions.

        Returns:
            pygame.Surface: The initialized display surface.
        """
        pygame.init()
        screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('2D Imitation Learning Scene')
        self.clock = pygame.time.Clock()

        return screen

    def add_agent(self, agent):
        """
        Adds an agent to the scene.

        Parameters:
            agent (Agent): The agent to be added to the scene.
        """
        self.agents.append(agent)

    def update_scene(self):
        """
        Updates and renders the scene including all agents.
        """
        self.screen.blit(self.background_image, (0, 0))

        for agent in self.agents:
            agent.render(self.screen)

        pygame.display.update()

    def run(self):
        """
        Main loop for updating the scene at a fixed rate.
        """
        self.update_scene()
        self.clock.tick(60)

    def take_screenshot(self, agent, width=200, height=200, offset_x=0, offset_y=0):
        """
        Takes a screenshot centered on the specified agent and rotated in the agent's direction.
        Returns the screenshot as a numpy array.

        Parameters:
            agent (Agent): The agent to center the screenshot on.
            width (int): Width of the screenshot.
            height (int): Height of the screenshot.
            offset_x (int): Horizontal offset from the center of the agent.
            offset_y (int): Vertical offset from the center of the agent.
        """
        screenshot_surface = pygame.Surface((width, height), pygame.SRCALPHA)

        screen_surface = self.screen.copy()

        angle = agent.angle + 90

        rotated_surface = pygame.transform.rotozoom(screen_surface, angle, 1.0)
        rotated_rect = rotated_surface.get_rect(center=(width // 2, height // 2))

        dx = self.screen_width // 2 - agent.x
        dy = self.screen_height // 2 - agent.y

        cos = math.cos(math.radians(-angle))
        sin = math.sin(math.radians(-angle))

        rotated_rect.x += dx * cos - dy * sin + offset_x
        rotated_rect.y += dx * sin + dy * cos + offset_y

        screenshot_surface.blit(rotated_surface, rotated_rect)

        return pygame.surfarray.array3d(screenshot_surface) / 255.0


class Agent:
    """
    Base class for an agent in the scene, capable of being rendered with a specific
    position and orientation.

    Attributes:
        image (pygame.Surface): Image representing the agent.
        x (int): x position of the agent.
        y (int): y position of the agent.
        angle (float): Orientation angle of the agent.
    """

    def __init__(self, image, x, y, angle):
        """
        Initializes the agent with its image, position, and orientation.

        Parameters:
            image (pygame.Surface): Image of the agent.
            x (int): Initial x position of the agent.
            y (int): Initial y position of the agent.
            angle (float): Initial orientation angle of the agent.
        """
        self.image = image
        self.image_width = image.get_width()
        self.image_height = image.get_height()
        self.x = x
        self.y = y
        self.angle = angle

    def render(self, screen):
        """
        Renders the agent on the specified screen.

        Parameters:
            screen (pygame.Surface): The display surface to render the agent on.
        """
        rotated_agent = pygame.transform.rotate(self.image, -self.angle)

        center = self.image.get_rect(topleft=(self.x, self.y)).center
        center = (center[0] - self.image_width / 2, center[1] - self.image_height / 2)
        agent_rect = rotated_agent.get_rect(center=center)

        screen.blit(rotated_agent, agent_rect)

    def update_position(self, x, y, angle):
        """
        Updates the position and orientation of the agent.

        Parameters:
            x (int): New x position of the agent.
            y (int): New y position of the agent.
            angle (float): New orientation angle of the agent.
        """
        self.x = x
        self.y = y
        self.angle = angle


class Car(Agent):
    """
    Represents a car agent in the scene with simple physics for movement and steering.

    Inherits from Agent and adds velocity, mass, and drag for basic physics simulation.
    """

    def __init__(self, x, y, angle, scale=1.0):
        """
        Initializes the car agent with position, orientation, and scale for its image.

        Parameters:
            x (int): Initial x position of the car.
            y (int): Initial y position of the car.
            angle (float): Initial orientation angle of the car.
            scale (float): Scale factor for the car image.
        """
        image = pygame.image.load('./scene_assets/car.png')
        image = pygame.transform.scale(image, (int(image.get_width() * scale), int(image.get_height() * scale)))

        super().__init__(image, x, y, angle)

        self.velocity = 0.0
        self.mass = 1200
        self.drag_coefficient = 8.0

    def update_physics(self, steer, throttle, brake):
        """
        Updates the car's physics based on throttle and steering input.

        Parameters:
            steer (float): Steering input, affecting orientation.
            throttle (float): Throttle input, affecting acceleration.
            brake (float): Brake input, affecting deceleration.
        """
        throttle_force = (throttle + .25) * 20
        drag_force = self.drag_coefficient * self.velocity
        net_force = throttle_force - drag_force - brake * 25
        acceleration = net_force / self.mass

        self.velocity += acceleration
        self.velocity = min(max(0.0, self.velocity), 1.0)

        self.angle += (steer * 1.5) * self.velocity
        self.angle = self.angle % 360

        direction = math.radians(self.angle - 90)
        dx = math.cos(direction) * self.velocity
        dy = math.sin(direction) * self.velocity

        self.x += dx
        self.y += dy

    def get_autopilot_control(self, scene, model):
        if model:
            image = scene.take_screenshot(self)
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image).float().unsqueeze(0)

            speed = torch.tensor([self.velocity], dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = model(image, speed)
                steer, throttle, brake = tuple(output.squeeze().tolist())

            return steer, throttle, brake
