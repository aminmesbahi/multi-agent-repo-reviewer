from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    sku: str
    name: str
    quantity: int

    def is_low(self, threshold: int = 5) -> bool:
        return self.quantity < threshold
