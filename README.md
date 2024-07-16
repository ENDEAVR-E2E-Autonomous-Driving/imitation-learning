[![Python Tests](https://github.com/SP24-CSCE482-capstone/github-classroom-setup-autonomous_driving_application_sprint_1/actions/workflows/python_tests.yml/badge.svg)](https://github.com/SP24-CSCE482-capstone/github-classroom-setup-autonomous_driving_application_sprint_1/actions/workflows/python_tests.yml)

# Imitation Learning: Autonomous Vehicles

This repository contains example code for training a neural network to drive a car in a 2D environment 
using imitation learning. It has been developed as part of a Senior Capstone project at Texas A&M University.
The project, at its core, is an exploration of the use of imitation learning in the context of autonomous vehicles.
This exploration has been developed for the [ENDEAVR Institute](https://endeavr.city/), a research institute at 
Texas A&M University under the guidance of [Dr. Wei Li](https://www.arch.tamu.edu/staff/wei-li/).

## Installation
The project is built using Python 3.8. We recommend utilizing Conda to ensure compatibility. For more information on
installing Conda, please refer to the [official documentation](https://docs.conda.io/en/latest/).
To install the required packages, run the following command:
```bash
pip install -e .
```

## Usage
The program consists of three primary components: data collection, model training, and model evaluation. To begin a new
data collection session, run the following command in either the cartoon or carla directory:

```bash
python main.py
```

### Data Collection
This will open a window displaying the 2D environment. The user can control the car using a variety of inputs (e.g. 
keyboard, joystick, etc.). If you have a game controller, the program will prompt you to map the controls to the
input. Otherwise, the program will default to using the arrow keys. A user can then drive the car around the environment.

The following keys can be used to control the program:
- `i`: Toggle between keyboard and joystick control, if a joystick is connected
- `r`: Reset the car to the starting position (carla)
- `q`: Quit the program
- `p`: Toggle autopilot
### Model Training

Training the model is done using the `train.py` script.

### Model Evaluation

In the carla directory, a topological planner is used from Carla 0.9.15 to generate a path for the car to follow. The
model is then used to predict the steering angle and throttle for the car to follow the path.

There are currently no automated ways to evaluate the model. The user must manually run the model and observe the
results.

## Authors

- [Preston Barnett](mailto:prestonb@tamu.edu)
- [Kirk Graham](mailto:kirk.jgraham3@tamu.edu)
- [Glenn Edstrom](mailto:glenn.edstrom@tamu.edu)
- [Isabelle Grimesey](mailto:isabelle.grimesey@tamu.edu)