import unittest
from unittest.mock import patch, mock_open
import imitation_shared.utils as utils


class UnitTestUtils(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    def test_load_config_success(self, mock_file):
        """Test loading a valid config file."""
        config = utils.load_config(mock_file)
        self.assertEqual(config, {"key": "value"})

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_config_failure(self, mock_file):
        """Test loading an invalid config file."""
        config = utils.load_config(mock_file)
        self.assertEqual(config, {})

    @patch("builtins.print")
    def test_print_formatted(self, mock_print):
        """Test the print_formatted function."""
        utils.print_formatted("Test message", color="\033[91m")
        mock_print.assert_called()

    @patch("builtins.print")
    def test_print_game_letterhead(self, mock_print):
        """Test the print_game_letterhead function."""
        utils.print_game_letterhead()
        self.assertTrue(mock_print.called)

    @patch('argparse.ArgumentParser.parse_args',
           return_value=utils.argparse.Namespace(sampling_rate=10, config='config.json'))
    def test_parse_args(self, _):
        """Test the parse_args function."""
        args = utils.parse_args()
        self.assertEqual(args.sampling_rate, 10)
        self.assertEqual(args.config, 'config.json')

    @patch("builtins.print")
    @patch('argparse.ArgumentParser.parse_args',
           return_value=utils.argparse.Namespace(sampling_rate=10, config='config.json'))
    def test_print_args(self, _, mock_print):
        """Test the print_args function with mock arguments."""
        args = utils.parse_args()
        utils.print_args(args)
        self.assertTrue(mock_print.called)
