import pytest
import pygame

from cartoon_simulation import scene

background_image = pygame.Surface((100, 100))
agent_image = pygame.Surface((10, 10))
initial_x = 0
initial_y = 0
initial_angle = 0


@pytest.fixture()
def scene_fixture():
    return scene.Scene(background_image, initial_x, initial_y, initial_angle)


@pytest.fixture()
def agent():
    agent_image.fill((255, 0, 0))
    return scene.Agent(agent_image, initial_x, initial_y, initial_angle)


def test_scene_with_agent(scene_fixture, agent):
    """Test the interaction between the Scene and Agent classes."""
    scene_fixture.add_agent(agent)
    scene_fixture.update_scene()
    assert scene_fixture.screen.get_at((0, 0)) == (255, 0, 0, 255)


def test_scene_with_moved_agent(scene_fixture, agent):
    """Test the interaction between the Scene and Agent classes by moving the agent."""
    scene_fixture.add_agent(agent)
    agent.x = 50
    scene_fixture.update_scene()
    assert scene_fixture.screen.get_at((0, 0)) == (0, 0, 0, 255)
    assert scene_fixture.screen.get_at((50, 0)) == (255, 0, 0, 255)
