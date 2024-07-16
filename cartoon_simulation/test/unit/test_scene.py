import pytest
import pygame

from cartoon_simulation import scene

background_image = pygame.Surface((100, 100))
initial_x = 0
initial_y = 0
initial_angle = 0


@pytest.fixture()
def scene_fixture():
    return scene.Scene(background_image, initial_x, initial_y, initial_angle)


def test_scene_init(scene_fixture):
    """Test the initialization of the Scene class."""
    assert scene_fixture.background_image == background_image
    assert scene_fixture.x == initial_x
    assert scene_fixture.y == initial_y
    assert scene_fixture.angle == initial_angle
    assert scene_fixture.screen_width == 100
    assert scene_fixture.screen_height == 100
    assert type(scene_fixture.clock) == pygame.time.Clock
    assert scene_fixture.agents == []
    assert type(scene_fixture.screen) == pygame.Surface


def test_update_screen(scene_fixture):
    """Test the update_screen method of the Scene class."""
    scene_fixture.update_scene()
    assert scene_fixture.screen.get_at((0, 0)) == (0, 0, 0, 255)
