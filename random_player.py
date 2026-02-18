from multiprocessing import Queue
from random import choice
from board import Entity, neighbors
import numpy as np
import numpy.typing as npt


def valid_neighbors(
    row: int, col: int, walls: npt.NDArray[np.int_]
) -> list[tuple[int, int]]:
    return [n for n in neighbors((row, col), walls.shape) if not walls[n]]


class RandomBot:

    def __init__(
        self,
        walls: npt.NDArray[np.int_],
        harvest_radius: int,
        vision_radius: int,
        battle_radius: int,
        max_turns: int,
        time_per_turn: float,
    ) -> None:
        self.walls = walls
        self.collect_radius = harvest_radius
        self.vision_radius = vision_radius
        self.battle_radius = battle_radius
        self.max_turns = max_turns
        self.time_per_turn = time_per_turn

    @property
    def name(self):
        return "rando"

    def move_ants(
        self,
        vision: set[tuple[tuple[int, int], Entity]],
        stored_food: int,
        move_queue: Queue,
    ):
        my_ants = {coord for coord, kind in vision if kind == Entity.FRIENDLY_ANT}
        my_hills = {coord for coord, kind in vision if kind == Entity.FRIENDLY_HILL}
        claimed_destinations = my_hills
        for ant in my_ants:
            valid = [
                v
                for v in valid_neighbors(*ant, self.walls)
                if v not in claimed_destinations
            ]
            if not valid:
                claimed_destinations.add(ant)
                continue
            dest = choice(valid)
            claimed_destinations.add(dest)
            move_queue.put((ant, dest))
