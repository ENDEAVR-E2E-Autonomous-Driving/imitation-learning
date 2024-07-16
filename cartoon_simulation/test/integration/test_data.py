import numpy as np

from cartoon_simulation.data import DataManager, ImitationDataset

def test_data_manager_and_loader(tmp_path):
    manager = DataManager(str(tmp_path), 'integration_test')
    image = np.random.rand(200, 200, 3)
    scalars = np.random.rand(1)
    targets = np.random.rand(3)

    manager.save(image, scalars, targets)
    manager.h5file.close()

    dataset = ImitationDataset(str(tmp_path))
    read_image, read_scalars, read_targets = dataset[0]

    assert np.array_equal(read_image, image.astype(np.float32).transpose((2, 0, 1)))
    assert np.array_equal(read_scalars, scalars.astype(np.float32))
    assert np.array_equal(read_targets, targets.astype(np.float32))