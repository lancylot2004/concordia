from abc import ABC
from abc import abstractmethod
from typing import Callable, Set, Tuple

from cleanup_cell import CleanupCell
from cleanup_cell import Empty
from cleanup_cell import Orchard
from cleanup_cell import River
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label

randomGen = np.random.default_rng(31)


class CleanupGridFactory(ABC):
    """A factory capable of generating a new grid for the Cleanup game."""

    @classmethod
    @abstractmethod
    def generate_grid(self) -> np.ndarray[CleanupCell]:
        pass


class SimpleCleanupGridFactory(CleanupGridFactory):
    """
    A factory which generates a simple grid for the Cleanup game. The grid will
    satisfy the following properties:
    - All non-river tiles will be reachable from all other non-river tiles.
    - (TODO) All river tiles will be "cleanable" by a player with the specified reach.
    """

    @classmethod
    def generate_grid(
        cls,
        width: int,
        height: int,
        pollution_rate: float,
        spawn_rate_func: Callable[[float], float] = lambda x: 1 / (x + 0.1),
        river_ratio: float = 0.2,
        river_width: int = 2,
        orchard_ratio: float = 0.1,
        reach: int = 3,
    ) -> np.ndarray[CleanupCell]:
        """_summary_

        Args:
            width (int): The width of the grid.
            height (int): The height of the grid.
            pollution_rate (float): The rate at which a particular river tile
                becomes polluted.
            spawn_rate_func (Callable[[float], float], optional): A function
                which given the current percentage of river tiles which are
                polluted, calculates the spawn rate of apples in orchard tiles.
                Defaults to `lambda x: 1 / (x + 0.1)`.
            river_ratio (float, optional): The approximate proportion of all
                tiles which should be river tiles. Defaults to 0.2.
            river_width (int, optional): The approximate width of the river.
                Defaults to 2.
            orchard_ratio (float, optional): The approximate proportion of all
                tiles which should be orchard tiles. Defaults to 0.1.
            reach (int, optional): The player's reach when cleaning pollution
                from the river. Defaults to 3.

        Returns:
            np.ndarray[CleanupCell]: The 2D array of [CleanupCell]s which
                represents the game grid.
        """

        assert 0 <= river_ratio <= 1
        assert 0 <= orchard_ratio <= 1
        assert 0 <= river_ratio + orchard_ratio <= 1

        # Initialise grid.
        grid = np.full((width, height), fill_value = Empty(), dtype = CleanupCell)

        # Generate river, and ensure the grid is still traversable.
        while True:
            river_points = SimpleCleanupGridFactory._generate_fuzzy_line(
                width = width, height = height,
                # Approximate length according to [river_ratio].
                length = int(width * height * river_ratio),
                reach = reach,
                thickness = river_width,
            )

            if SimpleCleanupGridFactory._ensure_traversable(width, height, river_points):
                break

        for (x, y) in river_points:
            grid[x, y] = River(pollution_rate = pollution_rate)

        # Calculate initial pollution.
        count_polluted, count_river = 0, 0
        for point in grid.flatten():
            if isinstance(point, River):
                count_river += 1
                if point.polluted: count_polluted += 1

        pollution = count_polluted / count_river

        # Generate orchard.
        orchard_points = SimpleCleanupGridFactory._generate_area(
            width = width, height = height,
            num_tiles = width * height * orchard_ratio,
            river_points = river_points
        )

        for (x, y) in orchard_points:
            grid[x, y] = Orchard(spawn_rate = spawn_rate_func(pollution))

        return grid

    @classmethod
    def _generate_fuzzy_line(
        cls,
        width: int,
        height: int,
        length: int,
        reach: int,
        thickness: int,
        fuzziness: int = 1
    ) -> Set[Tuple[int, int]]:
        """
        Generates the river (a fuzzy line) given the properties of the game
        grid, the length, thickness, and fuzziness of the river, and the
        player's reach.

        Args:
            width (int): The width of the grid.
            height (int): The height of the grid.
            length (int): Desired length of the river.
            reach (int): The player's maximum reach when cleaning pollution.
            thickness (int): Desired thickness of the river.
            fuzziness (int, optional): Arbitrary measure of how "messy" the
                river is. Defaults to 1.

        Returns:
            Set[Tuple[int, int]]: The set of points that compose the river.
        """

        # Calculate random starting point, and orient toward farther edge.
        startX, startY = (randomGen.integers(0, width - 1), randomGen.integers(0, height - 1))
        farthest_edge = np.argmax([startX, width - startX - 1, startY, height - startY - 1])

        target_points = [
            ((0, 0), (0, height - 1)),                  # Left edge
            ((width - 1, 0), (width - 1, height - 1)),  # Right edge
            ((0, 0), (width - 1, 0)),                   # Top edge
            ((0, height - 1), (width - 1, height - 1))  # Bottom edge
        ]

        target1, target2 = target_points[farthest_edge]
        angle1 = np.rad2deg(np.arctan2(target1[1] - startY, target1[0] - startX))
        angle2 = np.rad2deg(np.arctan2(target2[1] - startY, target2[0] - startX))

        if angle1 > angle2: angle1, angle2 = angle2, angle1
        angle = randomGen.uniform(angle1, angle2)

        # Generate the line with some... "fuzziness".
        anchor_points = set()
        for i in range(length):
            dx = int(np.cos(angle) * i)
            dy = int(np.sin(angle) * i)
            fx = startX + dx + randomGen.integers(-fuzziness, fuzziness + 1)
            fy = startY + dy + randomGen.integers(-fuzziness, fuzziness + 1)

            # Check for boundary overrun.
            if 0 <= fx < width and 0 <= fy < height:
                anchor_points.add((fx, fy))

        # Add thickness to line
        line_points = set()
        for fx, fy in anchor_points:
            for tx in range(-thickness // 2, (thickness + 1) // 2):
                for ty in range(-thickness // 2, (thickness + 1) // 2):
                    if 0 <= fx + tx < width and 0 <= fy + ty < height:
                        line_points.add((fx + tx, fy + ty))

        return line_points

    @classmethod
    def _generate_area(
        cls,
        width: int,
        height: int,
        num_tiles: int,
        river_points: Set[Tuple[int, int]]
    ) -> Set[Tuple[int, int]]:
        """
        Generates the orchard (a contiguous) area, attempting to place it as far
        away from the river as possible.

        Args:
            width (int): The width of the grid.
            height (int): The height of the grid.
            num_tiles (int): Desired number of tiles for the orchard.
            river_points (Set[Tuple[int, int]]): The set of points that
                constitute the river (which is non-traversable).

        Returns:
            Set[Tuple[int, int]]: The set of points that compose the orchard.
        """

        # Find spot farthest away from any river, to start flood-fill.
        grid = np.ones((height, width), dtype=bool)
        for (x, y) in river_points:
            grid[y, x] = False  # Mark river points as not available

        distances = distance_transform_edt(grid)
        farthest_point = np.unravel_index(np.argmax(distances), distances.shape)
        orchard_points = set()
        queue = [farthest_point]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while queue and len(orchard_points) < num_tiles:
            x, y = queue.pop(0)
            if (x, y) in orchard_points or (x, y) in river_points:
                continue

            orchard_points.add((x, y))
            randomGen.shuffle(directions)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and \
                    0 <= ny < width and \
                    (nx, ny) not in orchard_points and \
                    (nx, ny) not in river_points:
                    if randomGen.random() < 0.8:
                        queue.append((nx, ny))

        return orchard_points

    @classmethod
    def _ensure_traversable(
        cls,
        width: int,
        height: int,
        river_points: Set[Tuple[int, int]]
    ) -> bool:
        """
        In a grid of [width]x[height], ensure the line formed by [river_points]
        does not divide the grid into two or more separate components.

        Args:
            width (int): The width of the grid.
            height (int): The height of the grid.
            river_points (Set[Tuple[int, int]]): The set of points that
                constitute the river (which is non-traversable).

        Returns:
            bool: Whether traversability is satisfied.
        """

        # Create a binary grid where traversable (non-river) tiles are [True].
        binary_grid = np.zeros((width, height), dtype=bool)
        for (x, y) in river_points:
            binary_grid[x, y] = True

        labeled_grid, _ = label(binary_grid)
        component_sizes = np.bincount(labeled_grid.flat)
        non_river_components = component_sizes[1:]

        return len(non_river_components) == 1
