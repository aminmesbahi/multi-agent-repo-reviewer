import sys

from shelf_tracker.storage import load_items


def main() -> int:
    items = load_items(sys.argv[1])
    for item in items:
        flag = " (low)" if item.is_low() else ""
        print(f"{item.sku}: {item.name} x{item.quantity}{flag}")
    return 0
