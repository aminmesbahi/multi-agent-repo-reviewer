from shelf_tracker.models import Item


def test_is_low():
    assert Item("a1", "bolt", 2).is_low()
