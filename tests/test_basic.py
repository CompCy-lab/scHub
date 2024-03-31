import schub


def test_package_has_version():
    assert schub.__version__ is not None
