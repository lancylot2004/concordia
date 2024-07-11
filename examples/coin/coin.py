from datetime import datetime
from email import generator
from enum import Enum
from random import choice, sample
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


if __name__ == '__main__':
    grid = CoinGrid(5, 5)
    print(grid)
    print('\n')
    grid.move(CoinCell.RED, Direction.RIGHT)
    print(grid)

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
                   + self._grid.__str__()

    # Override
    def partial_state(self, player_name: str) -> str | None:
        # The coin game is fully observable for all players.
        return self.state()

    # Override
    def update(self) -> None:
        with self._lock:
            self._update_player_state("Red", self._red_memory)
            self._update_player_state("Blue", self._blue_memory)

    def _update_player_state(self, player_name: str, memory: AssociativeMemory) -> None:
        action = memory.retrieve_recent()
        prompt = InteractiveDocument(self._model)
        prompt.statement(f"Action: {action}\n")
        direction = prompt.open_question(
            f'Given the above action carried out by {player_name}, in',
            f'which direction did {player_name} move? Answer with "up"',
            f'"down", "left", or "right".',
        )

        self.move_player(player_name, direction)
        if self._verbose:
            print(prompt.view().text())

        self._history.append({
            'date': self._clock_call(),
            'state': self.state(),
            'player': player_name,
            'status': f'Moved {direction}',
        })

    # Override
    def terminate_episode(self) -> bool:
        return self._grid.blue_position == self._grid.coin_position \
               or self._grid.red_position == self._grid.coin_position

    def get_last_log(self):
        with self._lock:
            if self._history:
                return self._history[-1].copy()

    def move_player(self, player: str, direction: str) -> None:
        with self._lock:
            match player:
                case "Red": position = self._red_position
                case "Blue": position = self._blue_position
                case _: raise ValueError(f"Invalid player: {player}")

            x, y = position
            match direction.lower():
                case "up": y = max(0, y - 1)
                case "down": y = min(self._height - 1, y + 1)
                case "left": x = max(0, x - 1)
                case "right": x = min(self._width - 1, x + 1)
                case _: raise ValueError(f"Invalid direction: {direction}")

            self._grid[position] = CoinCell.EMPTY

            self._check_coin_reached()

    def _check_coin_reached(self) -> None:
        if self._red_position == self._coin_position:
            self._handle_coin_pickup("Red")
        if self._blue_position == self._coin_position:
            self._handle_coin_pickup("Blue")

    def _handle_coin_pickup(self, player: str) -> None:
        match (player, self._grid.coin_colour):
            case ("Red", CoinColour.RED): self._red_points += 1
            case ("Red", "blue"): self._red_points -= 2
            case ("Blue", "blue"): self._blue_points += 1
            case ("Blue", "red"): self._blue_points -= 2

        if player_name == "Red":
            if self._coin_color == "red":
                self._red_points += 1
            else:
                self._red_points += 1
                self._blue_points -= 2
        elif player_name == "Blue":
            if self._coin_color == "blue":
                self._blue_points += 1
            else:
                self._blue_points += 1
                self._red_points -= 2
