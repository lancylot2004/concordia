from typing import Tuple, override
from concordia.typing.component import Component
from cleanup import CleanupGame


class CleanupPlayer(Component):

    def __init__(
        self,
        sight: int,
        game: 'CleanupGame',
        location: Tuple[int, int]
    ) -> None:
        """Creates a new [Component] for a player of Cleanup.

        Args:
            sight (int): How many grid cells can the player see.
        """
        self._sight = sight
        self._game = game
        self._location = location

    @override
    def name(self) -> str:
        return "Cleanup game player"

    @override
    def state(self) -> str | None:
        return self._game.view()

    @override
    def partial_state(
        self,
        player_name: str,
    ) -> str | None:
        return self._game.partial_view(self._location, self._sight)
