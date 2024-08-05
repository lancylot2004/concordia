from typing import Tuple
import numpy as np
from examples.cleanup.cleanup_cell import CleanupCell
from examples.cleanup.grid_factory import CleanupGridFactory, SimpleCleanupGridFactory


class CleanupGame:
    """
    A mixed-motive multi-agent game where seven players coexist on a 2D grid
    world. There exists a river and an orchard area. The orchard area spawns
    apples at some rate 'r', which is inversely proportional to the overall
    pollution level in the river area.

    In each turn, players can choose one action from the set
    {move_{wasd}, pick_up, clean_up_{wasd}, zap_{wasd}}.
    """

    def __init__(
        self,
        grid: np.ndarray[CleanupCell],
        num_players: int,
    ) -> None:
        self._grid = grid

    def view(self) -> str:
        """Returns a string representation of the game state."""
        return ''.join([point.repr for point in line] for line in self._grid)

    def partial_view(self, center: Tuple[int, int], radius: int) -> str:
        """
        Returns a string representation of the game state around the center. All
        grid cells which cannot be seen are marked as 'X'.
        """
        centerX, centerY = center
        return ''.join(
            [point.repr if abs(centerX - x) <= radius and abs(centerY - y) <= radius else 'X'
            for x, point in enumerate(line)]
            for y, line in enumerate(self._grid)
        )
