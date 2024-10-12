import sys
import pytest
import unittest
from unittest.mock import patch
from click.testing import CliRunner
from importlib.metadata import PackageNotFoundError
import yada.cli
from yada.cli import run, _print_version
from yada.yada_cli import YadaCli


class TestCli(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
