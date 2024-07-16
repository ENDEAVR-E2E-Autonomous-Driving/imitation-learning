import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from cartoon_simulation.data import DataManager, ImitationDataset

# Unit Tests for DataManager
class TestDataManager:
    @pytest.fixture
    def manager(self, tmp_path):
        return DataManager(str(tmp_path), 'test')

    def test_initialize_folder(self, manager, tmp_path):
        assert os.path.exists(manager.folder)

    def test_file_path(self, manager):
        assert manager.file_path.startswith(manager.folder)
        assert manager.file_path.endswith('.h5')

    def test_open_file(self, manager):
        assert manager.h5file is not None

    def test_initialize_datasets(self, manager):
        for dataset in ['images', 'scalars', 'targets']:
            assert dataset in manager.h5file

    def test_save(self, manager):
        image = np.zeros((200, 200, 3))
        scalars = np.zeros((1,))
        targets = np.zeros((3,))
        manager.save(image, scalars, targets)

        for dataset in ['images', 'scalars', 'targets']:
            assert manager.h5file[dataset].shape[0] == 1

# Unit Tests for ImitationDataset
class TestImitationDataset:
    @pytest.fixture
    def manager(self, tmp_path):
        return DataManager(str(tmp_path), 'test')

    @pytest.fixture
    def dataset(self, tmp_path, manager):
        # Create some dummy data files
        manager.save(np.zeros((200, 200, 3)), [0], [0, 1, 2])
        manager.h5file.close()  # Close to simulate completed file writing
        return ImitationDataset(str(tmp_path))

    def test_file_paths(self, dataset, tmp_path):
        assert len(dataset.file_paths) > 0
        assert all(f.startswith(str(tmp_path)) for f in dataset.file_paths)

    def test_len(self, dataset):
        assert len(dataset) > 0

    def test_get_item(self, dataset):
        image, scalars, targets = dataset[0]
        assert image.shape == (3, 200, 200)
        assert scalars.shape == (1,)
        assert targets.shape == (3,)