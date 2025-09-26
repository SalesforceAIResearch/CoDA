import unittest


class TestImportTorchax(unittest.TestCase):
  def test_import_torchax(self):
    """
    This test checks that `torchax` is installed.
    """
    import torchax  # noqa: F401
