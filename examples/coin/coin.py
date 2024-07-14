from datetime import datetime
from email import generator
from enum import Enum
from random import choice, sample
from re import search
from threading import Lock
from typing import Callable

from concordia.associative_memory.associative_memory import AssociativeMemory
from concordia.document.interactive_document import InteractiveDocument
from concordia.language_model.language_model import LanguageModel
from concordia.typing.component import Component

class CoinCell(Enum):
    RED = 'R'
    BLUE = 'B'
    COIN = 'C'
    EMPTY = '.'

class CoinColour(Enum):
    RED = 'red'
    BLUE = 'blue'

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class CoinGrid:
    def __init__(self, width: int, height: int) -> None:
        assert width > 0 and height > 0

        self._width, self._height = width, height
        self.new_round()

    def __str__(self) -> str:
        return '\n'.join(''.join(self._get_cell_repr(x, y) for x in range(self._width)) for y in range(self._height))

    def _get_cell_repr(self, x, y) -> str:
        for cell_type, pos in self._grid.items():
            if pos == (x, y):
                return cell_type.value
        return CoinCell.EMPTY.value

    def new_round(self) -> None:
        red, blue, coin = sample([(x, y) for x in range(self._width) for y in range(self._height)], 3)
        self._grid = {
            CoinCell.RED: red,
            CoinCell.BLUE: blue,
            CoinCell.COIN: coin,
        }
        self._coin_colour = choice([e.value for e in CoinColour])

    def move(self, cell: CoinCell, direction: Direction) -> None:
        assert cell in (CoinCell.RED, CoinCell.BLUE), "Only players can move!"

        x, y = self._grid[cell]
        dx, dy = direction.value
        new_x, new_y = x + dx, y + dy

        assert 0 <= new_x < self._width and 0 <= new_y < self._height, "Out of bound movement!"
        self._grid[cell] = (new_x, new_y)

    @property
    def red_position(self) -> tuple:
        return self._grid[CoinCell.RED]

    @property
    def blue_position(self) -> tuple:
        return self._grid[CoinCell.BLUE]

    @property
    def coin_position(self) -> tuple:
        return self._grid[CoinCell.COIN]

    @property
    def coin_colour(self) -> str:
        return self._coin_colour

"""Tracks the current state of the coin game."""
class CoinState(Component):
    def __init__(
        self,
        clock_now: Callable[[], datetime],
        model: LanguageModel,
        red_memory: AssociativeMemory,
        blue_memory: AssociativeMemory,
        width: int,
        height: int,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self._lock = Lock()
        self._clock_call = clock_now
        self._model = model
        self._red_points, self._blue_points = 0, 0
        self._red_memory, self._blue_memory = red_memory, blue_memory
        self._width, self._height = width, height
        self._grid = CoinGrid(width, height)
        self._verbose = verbose
        self._history = []

    # Override
    def name(self) -> str:
        return "State of Coin Game"

    # Override
    def state(self) -> str:
        with self._lock:
            return f"Red: {self._red_points}, Blue: {self._blue_points}, Grid: {self._width}x{self._height}\n" \
                   + self._grid.__str__() + '\n'

    # Override
    def partial_state(self, player_name: str) -> str | None:
        # The coin game is fully observable for all players.
        return self.state()

    def update_after_event(self, event_statement: str) -> None:
        with self._lock:
            self._update_player(CoinCell.RED, self._red_memory)
            self._update_player(CoinCell.BLUE, self._blue_memory)

    def _update_player(self, player: CoinCell, memory: AssociativeMemory) -> None:
        action = memory.retrieve_by_regex(r".", sort_by_time = True)[-1]
        direction = search(r"(up|down|left|right)", action.lower())
        if direction is None: return

        match direction.group(0):
            case "up": direction = Direction.UP
            case "down": direction = Direction.DOWN
            case "left": direction = Direction.LEFT
            case "right": direction = Direction.RIGHT
        assert direction in Direction, f"Invalid direction {direction}!"

        self._grid.move(player, direction)
        if self._verbose:
            print(f"Player {player.value} moved {direction}!")
        self._history.append({
            'date': self._clock_call(),
            'state': self.state(),
            'player': player.value,
            'status': f'Moved {direction}',
        })

    def _update_game(self) -> None:
        if not self.terminate_episode(): return

        if self._grid.blue_position == self._grid.coin_position:
            self._blue_points += 1 if self._grid.coin_colour == CoinColour.BLUE else -2
        if self._grid.red_position == self._grid.coin_position:
            self._red_points += 1 if self._grid.coin_colour == CoinColour.RED else -2

        self._grid.new_round()

    # Override
    def terminate_episode(self) -> bool:
        return self._grid.blue_position == self._grid.coin_position \
               or self._grid.red_position == self._grid.coin_position

    def get_last_log(self):
        with self._lock:
            if self._history:
                return self._history[-1].copy()
