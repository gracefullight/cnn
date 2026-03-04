# ruff: noqa: N999
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def sigmoid(z: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Return the sigmoid activation value."""
    z_array = np.asarray(z, dtype=np.float64)
    result = 1.0 / (1.0 + np.exp(-z_array))

    if np.isscalar(z):
        return float(result)
    return result


def loss(a: float, y: float, eps: float = 1e-12) -> float:
    """Calculate binary negative log-likelihood loss."""
    a_clipped = float(np.clip(a, eps, 1.0 - eps))
    return float(-(y * np.log(a_clipped) + (1.0 - y) * np.log(1.0 - a_clipped)))


def main() -> None:
    # Step 1) Sigmoid 테스트
    z = 0.458
    print(f"sigmoid({z}) = {sigmoid(z)}")  # noqa: T201

    # Step 2) Log loss 테스트
    print(f"loss(0.7, 1) = {loss(0.7, 1)}")  # noqa: T201


if __name__ == "__main__":
    main()
