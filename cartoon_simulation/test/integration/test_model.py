from cartoon_simulation.model import CNNModel, save_model, load_model

def test_model_save(tmp_path):
    model = CNNModel()

    model_name = 'integration_test_model'
    save_model(str(tmp_path), model, model_name)

    loaded_model = load_model(str(tmp_path), model_name)

    assert model.state_dict().keys() == loaded_model.state_dict().keys()
