import platform
from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


package_name = "imitation_learning"
version = "0.1"


def setup_packages():
    if platform.system() == "Windows":
        requirements.append("torch")
        requirements.append("torchvision")
        requirements.append("carla")

        setup(
            name=package_name,
            version=version,
            packages=['carla_simulation', 'cartoon_simulation', 'imitation_shared'],
            install_requires=requirements,
        )
    elif platform.system() == "Darwin":
        requirements.append("torch")
        requirements.append("torchvision")

        # CARLA not supported on macOS
        setup(
            name=package_name,
            version=version,
            packages=['cartoon_simulation', 'imitation_shared'],
            install_requires=requirements,
        )
    else:
        requirements.append("torch")
        requirements.append("torchvision")
        requirements.append("carla")

        setup(
            name=package_name,
            version=version,
            packages=['carla_simulation', 'cartoon_simulation', 'imitation_shared'],
            install_requires=requirements,
        )


setup_packages()
