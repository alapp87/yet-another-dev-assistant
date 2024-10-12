import unittest
from unittest.mock import patch, call, MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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

    @patch("yada.yada_cli.YadaAgent.invoke")
    @patch("yada.yada_cli.YadaCli._handle_event")
    @patch("yada.yada_cli.YadaCli._handle_tool_calls")
    def test_yada_command(self, mock_handle_tool_calls, mock_handle_event, mock_invoke):
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
        mock_handle_event.assert_called_once_with(mock_result)
        mock_handle_tool_calls.assert_called_once_with(mock_result)

    @patch("yada.yada_cli.utils.agent_response")
    @patch("yada.yada_cli.utils.user_input", side_effect=["q"])
    @patch("yada.yada_cli.utils.say_goodbye")
    @patch("yada.yada_cli.YadaCli._print_title")
    def test_yada_chat_exit(
        self,
        mock_print_title,
        mock_say_goodbye,
        mock_user_input,
        mock_agent_response,
    ):
        # Act
        self.yada_cli.yada_chat()

        # Assert
        mock_print_title.assert_called_once()
        mock_agent_response.assert_called_once_with("Hello! How can I help you?")
        mock_user_input.assert_called_once()
        mock_say_goodbye.assert_called_once()

    @patch("yada.yada_cli.utils.agent_response")
    @patch("yada.yada_cli.utils.user_input", side_effect=["", "test prompt", "q"])
    @patch("yada.yada_cli.utils.print_thinking")
    @patch("yada.yada_cli.utils.say_goodbye")
    @patch("yada.yada_cli.YadaCli._print_title")
    @patch("yada.yada_cli.YadaCli._handle_event")
    @patch("yada.yada_cli.YadaCli._handle_tool_calls")
    @patch(
        "yada.yada_cli.YadaAgent.stream",
        return_value=[{"messages": [{"id": "123", "content": "test response"}]}],
    )
    def test_yada_chat_interaction(
        self,
        mock_stream,
        mock_handle_tool_calls,
        mock_handle_event,
        mock_print_title,
        mock_say_goodbye,
        mock_print_thinking,
        mock_user_input,
        mock_agent_response,
    ):
        # Act
        self.yada_cli.yada_chat()

        # Assert
        mock_print_title.assert_called_once()
        mock_agent_response.assert_called_once_with("Hello! How can I help you?")
        self.assertEqual(mock_user_input.call_count, 3)
        mock_print_thinking.assert_called_once()
        mock_stream.assert_called_once_with(
            {"messages": ["test prompt"]}, config=self.yada_cli.config
        )
        mock_handle_event.assert_called_once_with(
            {"messages": [{"id": "123", "content": "test response"}]}
        )
        mock_handle_tool_calls.assert_called_once_with(
            {"messages": [{"id": "123", "content": "test response"}]}
        )
        mock_say_goodbye.assert_called_once()

    @patch("yada.yada_cli.ToolLoader")
    @patch("yada.yada_cli.YadaAgent")
    @patch("yada.yada_cli.model")
    @patch("yada.yada_cli.MemorySaver")
    def test_new_agent(
        self, mock_memory_saver, mock_model, mock_yada_agent, mock_tool_loader
    ):
        # Arrange
        mock_tool_loader_instance = MagicMock()
        mock_tool_loader.return_value = mock_tool_loader_instance
        mock_tool_loader_instance.safe_tools = ["safe_tool"]
        mock_tool_loader_instance.sensitive_tools = ["sensitive_tool"]

        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        mock_memory_saver_instance = MagicMock()
        mock_memory_saver.return_value = mock_memory_saver_instance

        # Act
        agent = self.yada_cli._new_agent()

        # Assert
        mock_tool_loader_instance.load.assert_called_once()
        mock_yada_agent.assert_called_once_with(
            model=mock_model_instance,
            safe_tools=["safe_tool"],
            sensitive_tools=["sensitive_tool"],
            checkpointer=mock_memory_saver_instance,
            debug=self.yada_cli.debug,
        )
        self.assertEqual(agent, mock_yada_agent.return_value)

    @patch("yada.yada_cli.utils.user_response")
    @patch("yada.yada_cli.YadaCli._handle_ai_message")
    def test_handle_event_with_human_message(
        self, mock_handle_ai_message, mock_user_response
    ):
        # Arrange
        event = {
            "dialog_state": ["state1", "state2"],
            "messages": HumanMessage(id="123", content="test human message"),
        }
        self.yada_cli._printed = set()

        # Act
        self.yada_cli._handle_event(event, print_user_events=True)

        # Assert
        mock_user_response.assert_called_once_with("test human message")
        mock_handle_ai_message.assert_not_called()
        self.assertIn("123", self.yada_cli._printed)

    @patch("yada.yada_cli.utils.user_response")
    @patch("yada.yada_cli.YadaCli._handle_ai_message")
    def test_handle_event_with_ai_message(
        self, mock_handle_ai_message, mock_user_response
    ):
        # Arrange
        event = {
            "dialog_state": ["state1", "state2"],
            "messages": AIMessage(id="123", content="test ai message"),
        }
        self.yada_cli._printed = set()

        # Act
        self.yada_cli._handle_event(event, print_user_events=False)

        # Assert
        mock_user_response.assert_not_called()
        mock_handle_ai_message.assert_called_once_with(event["messages"])
        self.assertIn("123", self.yada_cli._printed)

    @patch("yada.yada_cli.utils.user_response")
    @patch("yada.yada_cli.YadaCli._handle_ai_message")
    def test_handle_event_with_already_printed_message(
        self, mock_handle_ai_message, mock_user_response
    ):
        # Arrange
        event = {
            "dialog_state": ["state1", "state2"],
            "messages": AIMessage(id="123", content="test ai message"),
        }
        self.yada_cli._printed = {"123"}

        # Act
        self.yada_cli._handle_event(event, print_user_events=False)

        # Assert
        mock_user_response.assert_not_called()
        mock_handle_ai_message.assert_not_called()

    @patch("yada.yada_cli.utils.user_response")
    @patch("yada.yada_cli.YadaCli._handle_ai_message")
    def test_handle_event_with_no_dialog_state(
        self, mock_handle_ai_message, mock_user_response
    ):
        # Arrange
        event = {"messages": AIMessage(id="123", content="test ai message")}
        self.yada_cli._printed = set()

        # Act
        self.yada_cli._handle_event(event, print_user_events=False)

        # Assert
        mock_user_response.assert_not_called()
        mock_handle_ai_message.assert_called_once_with(event["messages"])
        self.assertIn("123", self.yada_cli._printed)

    @patch("yada.yada_cli.utils.user_input", side_effect=["y", "n"])
    @patch("yada.yada_cli.YadaCli._handle_user_response_to_sensitive_tool_call")
    @patch("yada.yada_cli.YadaCli._handle_ai_message")
    @patch("yada.yada_cli.YadaAgent.get_state")
    def test_handle_tool_calls(
        self,
        mock_get_state,
        mock_handle_ai_message,
        mock_handle_user_response_to_sensitive_tool_call,
        mock_user_input,
    ):
        # Arrange
        event = {"messages": [{"id": "123", "content": "test response"}]}
        mock_snapshot = MagicMock()
        mock_snapshot.next = True
        mock_get_state.side_effect = [
            mock_snapshot,
            mock_snapshot,
            MagicMock(next=False),
        ]
        mock_result = {"messages": [{"id": "456", "content": "test ai message"}]}
        mock_handle_user_response_to_sensitive_tool_call.return_value = mock_result

        # Act
        self.yada_cli._handle_tool_calls(event)

        # Assert
        self.assertEqual(mock_user_input.call_count, 2)
        mock_handle_user_response_to_sensitive_tool_call.assert_has_calls(
            [call("y", event), call("n", event)]
        )
        mock_handle_ai_message.assert_called_with(mock_result["messages"][-1])
        self.assertEqual(mock_get_state.call_count, 3)

    @patch("yada.yada_cli.utils.print_working")
    @patch("yada.yada_cli.YadaAgent.invoke")
    def test_handle_user_response_to_sensitive_tool_call_yes(
        self, mock_invoke, mock_print_working
    ):
        # Arrange
        user_prompt = "y"
        event = {
            "messages": [
                AIMessage(
                    id="123",
                    content="test ai message",
                    tool_calls=[
                        {"id": "tool1", "name": "tool1", "args": {"arg1": "value1"}}
                    ],
                )
            ]
        }
        mock_result = {"messages": [{"id": "456", "content": "test ai message"}]}
        mock_invoke.return_value = mock_result

        # Act
        result = self.yada_cli._handle_user_response_to_sensitive_tool_call(
            user_prompt, event
        )

        # Assert
        mock_print_working.assert_called_once()
        mock_invoke.assert_called_once_with(None, self.yada_cli.config)
        self.assertEqual(result, mock_result)

    @patch("yada.yada_cli.utils.print_thinking")
    @patch("yada.yada_cli.YadaAgent.invoke")
    def test_handle_user_response_to_sensitive_tool_call_no(
        self, mock_invoke, mock_print_thinking
    ):
        # Arrange
        user_prompt = "n"
        event = {
            "messages": [
                AIMessage(
                    id="123",
                    content="test ai message",
                    tool_calls=[{"id": "tool1", "name": "tool1", "args": {}}],
                )
            ]
        }
        mock_result = {"messages": [{"id": "456", "content": "test ai message"}]}
        mock_invoke.return_value = mock_result

        # Act
        result = self.yada_cli._handle_user_response_to_sensitive_tool_call(
            user_prompt, event
        )

        # Assert
        mock_print_thinking.assert_called_once()
        mock_invoke.assert_called_once_with(
            {
                "messages": [
                    ToolMessage(
                        tool_call_id="tool1",
                        content="Tool call denied by user. Reasoning: 'No, I don't want to execute those tools.'. Continue assisting, accounting for the user's input.",
                    )
                ]
            },
            self.yada_cli.config,
        )
        self.assertEqual(result, mock_result)

    @patch("yada.yada_cli.utils.print_thinking")
    @patch("yada.yada_cli.YadaAgent.invoke")
    def test_handle_user_response_to_sensitive_tool_call_custom_reason(
        self, mock_invoke, mock_print_thinking
    ):
        # Arrange
        user_prompt = "I need more information."
        event = {
            "messages": [
                AIMessage(
                    id="123",
                    content="test ai message",
                    tool_calls=[{"id": "tool1", "name": "tool1", "args": {}}],
                )
            ]
        }
        mock_result = {"messages": [{"id": "456", "content": "test ai message"}]}
        mock_invoke.return_value = mock_result

        # Act
        result = self.yada_cli._handle_user_response_to_sensitive_tool_call(
            user_prompt, event
        )

        # Assert
        mock_print_thinking.assert_called_once()
        mock_invoke.assert_called_once_with(
            {
                "messages": [
                    ToolMessage(
                        tool_call_id="tool1",
                        content="Tool call denied by user. Reasoning: 'I need more information.'. Continue assisting, accounting for the user's input.",
                    )
                ]
            },
            self.yada_cli.config,
        )
        self.assertEqual(result, mock_result)

    @patch("yada.yada_cli.utils.print_working")
    @patch("yada.yada_cli.utils.agent_response")
    @patch("yada.yada_cli.YadaCli._print_tool_calls_message")
    @patch("yada.yada_cli.YadaAgent.is_sensitive_tool_call_exist")
    def test_handle_ai_message_with_sensitive_tool_call(
        self,
        mock_is_sensitive_tool_call_exist,
        mock_print_tool_calls_message,
        mock_agent_response,
        mock_print_working,
    ):
        # Arrange
        message = AIMessage(
            id="123",
            content="test ai message",
            tool_calls=[{"id": "tool1", "name": "tool1", "args": {}}],
        )
        mock_is_sensitive_tool_call_exist.return_value = True

        # Act
        self.yada_cli._handle_ai_message(message)

        # Assert
        mock_is_sensitive_tool_call_exist.assert_called_once_with(message.tool_calls)
        mock_print_tool_calls_message.assert_called_once_with(message.tool_calls)
        mock_print_working.assert_not_called()
        mock_agent_response.assert_not_called()

    @patch("yada.yada_cli.utils.print_working")
    @patch("yada.yada_cli.utils.agent_response")
    @patch("yada.yada_cli.YadaCli._print_tool_calls_message")
    @patch("yada.yada_cli.YadaAgent.is_sensitive_tool_call_exist")
    def test_handle_ai_message_with_safe_tool_call(
        self,
        mock_is_sensitive_tool_call_exist,
        mock_print_tool_calls_message,
        mock_agent_response,
        mock_print_working,
    ):
        # Arrange
        message = AIMessage(
            id="123",
            content="test ai message",
            tool_calls=[{"id": "tool1", "name": "tool1", "args": {}}],
        )
        mock_is_sensitive_tool_call_exist.return_value = False

        # Act
        self.yada_cli._handle_ai_message(message)

        # Assert
        mock_is_sensitive_tool_call_exist.assert_called_once_with(message.tool_calls)
        mock_print_tool_calls_message.assert_not_called()
        mock_print_working.assert_called_once()
        mock_agent_response.assert_not_called()

    @patch("yada.yada_cli.utils.print_working")
    @patch("yada.yada_cli.utils.agent_response")
    @patch("yada.yada_cli.YadaCli._print_tool_calls_message")
    @patch("yada.yada_cli.YadaAgent.is_sensitive_tool_call_exist")
    def test_handle_ai_message_with_no_tool_calls(
        self,
        mock_is_sensitive_tool_call_exist,
        mock_print_tool_calls_message,
        mock_agent_response,
        mock_print_working,
    ):
        # Arrange
        message = AIMessage(id="123", content="test ai message", tool_calls=[])

        # Act
        self.yada_cli._handle_ai_message(message)

        # Assert
        mock_is_sensitive_tool_call_exist.assert_not_called()
        mock_print_tool_calls_message.assert_not_called()
        mock_print_working.assert_not_called()
        mock_agent_response.assert_called_once_with(message.content)

    @patch("yada.yada_cli.utils.agent_response")
    def test_print_tool_calls_message_single_tool(self, mock_agent_response):
        # Arrange
        tool_calls = [
            {
                "id": "tool1",
                "name": "Tool 1",
                "args": {"arg1": "value1", "arg2": "value2"},
            }
        ]

        # Act
        self.yada_cli._print_tool_calls_message(tool_calls)

        # Assert
        expected_message = (
            "I want to execute the following tools. Reply 'y' to continue or 'n' to cancel. Otherwise you can explain your requested changes."
            "\n\n**Calling tool(s)**\n"
            "- **Tool:** Tool 1\n\t- **Args**\n"
            "\t\t- arg1=value1\n"
            "\t\t- arg2=value2\n"
        )
        mock_agent_response.assert_called_once_with(expected_message)

    @patch("yada.yada_cli.utils.agent_response")
    def test_print_tool_calls_message_multiple_tools(self, mock_agent_response):
        # Arrange
        tool_calls = [
            {
                "id": "tool1",
                "name": "Tool 1",
                "args": {"arg1": "value1"},
            },
            {
                "id": "tool2",
                "name": "Tool 2",
                "args": {"argA": "valueA", "argB": "valueB"},
            },
        ]

        # Act
        self.yada_cli._print_tool_calls_message(tool_calls)

        # Assert
        expected_message = (
            "I want to execute the following tools. Reply 'y' to continue or 'n' to cancel. Otherwise you can explain your requested changes."
            "\n\n**Calling tool(s)**\n"
            "- **Tool:** Tool 1\n\t- **Args**\n"
            "\t\t- arg1=value1\n"
            "- **Tool:** Tool 2\n\t- **Args**\n"
            "\t\t- argA=valueA\n"
            "\t\t- argB=valueB\n"
        )
        mock_agent_response.assert_called_once_with(expected_message)

    @patch("yada.yada_cli.utils.agent_response")
    def test_print_tool_calls_message_no_args(self, mock_agent_response):
        # Arrange
        tool_calls = [
            {
                "id": "tool1",
                "name": "Tool 1",
                "args": {},
            }
        ]

        # Act
        self.yada_cli._print_tool_calls_message(tool_calls)

        # Assert
        expected_message = (
            "I want to execute the following tools. Reply 'y' to continue or 'n' to cancel. Otherwise you can explain your requested changes."
            "\n\n**Calling tool(s)**\n"
            "- **Tool:** Tool 1\n\t- **Args**\n"
        )
        mock_agent_response.assert_called_once_with(expected_message)
