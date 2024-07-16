import pytest
import torch

from cartoon_simulation.model import CNNModel, save_model, load_model

width = 200
height = 200
outputs = 3

class TestCNNModel:
    @pytest.fixture
    def model(self):
        return CNNModel()

    def test_forward_pass(self, model):
        x_img = torch.rand(4, 3, height, width)
        scalar = torch.rand(4, 1)

        output = model(x_img, scalar)

        assert output.shape == (4, outputs)

class TestModelSavingLoading:
    @pytest.fixture
    def model(self):
        return CNNModel()

    def test_save_load_model(self, tmp_path, model):
        model_name = 'test_model'
        save_model(str(tmp_path), model, model_name)

        saved_model_path = tmp_path / f"{model_name}.pth"
        assert saved_model_path.exists()

        loaded_model = load_model(str(tmp_path), model_name)

        assert model.state_dict().keys() == loaded_model.state_dict().keys()
