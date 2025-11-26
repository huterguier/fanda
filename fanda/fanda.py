from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt

@dataclass
class Fanda:
    fig: Optional[plt.Figure] = None
    ax: Optional[plt.Axes] = None

    def pipe(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)
