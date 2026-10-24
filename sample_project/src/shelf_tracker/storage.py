import json

from shelf_tracker.models import Item


def load_items(path):
    with open(path) as handle:
        raw = json.load(handle)
    return [Item(**entry) for entry in raw]


def save_items(path, items):
    payload = [{"sku": i.sku, "name": i.name, "quantity": i.quantity} for i in items]
    try:
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)
    except OSError as exc:
        raise RuntimeError(f"could not save inventory to {path}") from exc
