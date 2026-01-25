from bdmtx.__main__ import main


def test_smoke():
    """Smoke test main entrypoint."""
    assert callable(main)
