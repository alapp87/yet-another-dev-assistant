import unittest
from unittest.mock import patch, MagicMock

from yada.yada_cli import YadaCli


class TestYadaCli(unittest.TestCase):
    def setUp(self) -> None:
        thread_id = None
        debug = False
        self.yada_cli = YadaCli(thread_id=thread_id, debug=debug)
        return super().setUp()

    @patch("yada.yada_cli.YadaAgent.invoke")
    @patch("yada.yada_cli.YadaCli._print_event")
    @patch("yada.yada_cli.YadaCli._handle_tool_calls")
    def test_yada_command(self, mock_handle_tool_calls, mock_print_event, mock_invoke):
        # Arrange
        command = "test command"
        mock_result = {"messages": [{"id": "123", "content": "test response"}]}
        mock_invoke.return_value = mock_result

        # Act
        self.yada_cli.yada_command(command)

        # Assert
        mock_invoke.assert_called_once_with(
            {"messages": [command]}, config=self.yada_cli.config
        )
        mock_print_event.assert_called_once_with(mock_result, self.yada_cli._printed)
        mock_handle_tool_calls.assert_called_once_with(
            mock_result, self.yada_cli.config
        )
