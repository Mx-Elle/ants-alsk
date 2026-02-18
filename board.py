from enum import Enum
from functools import cache
import sys

import pygame
import pygame.locals
from dataclasses import dataclass
from itertools import combinations, product
from random import shuffle
from typing import Iterable, Optional
import numpy as np
import numpy.typing as npt
from scipy.signal import convolve2d


class Entity(Enum):

    FRIENDLY_HILL = 1
    ENEMY_HILL = 2
    FRIENDLY_ANT = 3
    ENEMY_ANT = 4
    FOOD = 5


@dataclass
class Board:

    walls: npt.NDArray[np.int_]
    hills: npt.NDArray[np.int_]

    def __post_init__(self):
        assert self.walls.shape == self.hills.shape
        self.ants = np.zeros(self.walls.shape).astype(int)
        self.food = np.zeros(self.walls.shape).astype(int)
        self.food_spawn_order = [
            (r, c)
            for r, c in zip(*np.where(self.walls == 0))
            if r >= c and self.hills[r, c] == 0
        ]
        shuffle(self.food_spawn_order)
        self.food_spawn_index = 0

    def can_see(
        self,
        locations: Iterable[tuple[int, int]],
        vision_range_2: int,
        target: tuple[int, int],
    ):
        for loc in locations:
            if toroidal_distance_2(target, loc, self.shape) <= vision_range_2:
                return True
        return False

    def get_vision(
        self, player: int, vision_range: int
    ) -> set[tuple[tuple[int, int], Entity]]:
        vis_2 = vision_range**2
        ant_locs = set(zip(*np.where(self.ants == player)))
        hill_locs = set(zip(*np.where(self.hills == player)))
        my_locs = ant_locs | hill_locs
        food_locs = {
            loc
            for loc in zip(*np.where(self.food))
            if self.can_see(my_locs, vis_2, loc)
        }
        enemy_locs = {
            loc
            for loc in zip(*np.where(self.ants == 3 - player))
            if self.can_see(my_locs, vis_2, loc)
        }
        enemy_hill_locs = set(zip(*np.where(self.ants == 3 - player)))
        return (
            {(loc, Entity.FRIENDLY_ANT) for loc in ant_locs}
            | {(loc, Entity.FRIENDLY_HILL) for loc in hill_locs}
            | {(loc, Entity.FOOD) for loc in food_locs}
            | {(loc, Entity.ENEMY_ANT) for loc in enemy_locs}
            | {(loc, Entity.ENEMY_HILL) for loc in enemy_hill_locs}
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.walls.shape

    def wrap(self, coord: tuple[int, int]) -> tuple[int, int]:
        return (coord[0] % self.shape[0], coord[1] % self.shape[1])

    def mirror(self, coord: tuple[int, int]):
        return (self.shape[0] - coord[0] - 1, self.shape[1] - coord[1] - 1)

    def spawn_food(self) -> None:
        food_coord = self.food_spawn_order[self.food_spawn_index]
        self.food[food_coord] = 1
        self.food[self.mirror(food_coord)] = 1
        self.food_spawn_index += 1
        if self.food_spawn_index >= len(self.food_spawn_order):
            self.food_spawn_index = 0
            shuffle(self.food_spawn_order)

    def render(self, width: int, height: int, padding: float = 0.15) -> pygame.Surface:
        pad_rows, pad_cols = int(self.shape[0] * padding), int(self.shape[1] * padding)
        map_surface = pygame.Surface((width, height))
        cell_w, cell_h = width / (self.shape[1] + 2 * pad_cols), height / (
            self.shape[0] + 2 * pad_rows
        )
        for row, col in product(
            range(-pad_rows, self.shape[0] + pad_rows),
            range(-pad_cols, self.shape[1] + pad_cols),
        ):
            x, y = (col + pad_cols) * cell_w, (row + pad_rows) * cell_h
            pad_cell = False
            pad_cell = row not in range(self.shape[0]) or col not in range(
                self.shape[1]
            )
            row, col = row % self.shape[0], col % self.shape[0]

            palette = (
                [
                    "#772d00",
                    "#ffffff",
                    "#56c235",
                    "#ff5100",
                    "#a200ff",
                    "#ff0000",
                    "#0000ff",
                ]
                if not pad_cell
                else [
                    "#3c1700",
                    "#8a8a8a",
                    "#316f1e",
                    "#9e3200",
                    "#570089",
                    "#8b0000",
                    "#000089",
                ]
            )

            pygame.draw.rect(
                map_surface,
                palette[0] if self.walls[row, col] else palette[1],
                (x, y, cell_w + 1, cell_h + 1),
            )
            if self.food[row, col] == 1:
                pygame.draw.rect(
                    map_surface,
                    palette[2],
                    (x, y, cell_w + 1, cell_h + 1),
                )
            if self.hills[row, col] != 0:
                pygame.draw.circle(
                    map_surface,
                    palette[3] if self.hills[row, col] == 1 else palette[4],
                    (x + cell_w / 2, y + cell_h / 2),
                    min(cell_w, cell_h) / 2,
                    3,
                )
            if self.ants[row, col] != 0:
                pygame.draw.circle(
                    map_surface,
                    palette[5] if self.ants[row, col] == 1 else palette[6],
                    (x + cell_w / 2, y + cell_h / 2),
                    0.6 * min(cell_w, cell_h) / 2,
                )
        return map_surface


def neighbors(
    loc: tuple[int, int], shape: tuple[int, int]
) -> Iterable[tuple[int, int]]:
    for dr, dc in {(-1, 0), (1, 0), (0, -1), (0, 1)}:
        yield (loc[0] + dr) % shape[0], (loc[1] + dc) % shape[1]


@cache
def toroidal_distance_2(
    a: tuple[int, int], b: tuple[int, int], shape: tuple[int, int]
) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    if dr > 0.5 * shape[0]:
        dr = shape[0] - dr
    if dc > 0.5 * shape[1]:
        dc = shape[1] - dc
    return dr**2 + dc**2


def _segment(walls: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    seg = 1
    segments = np.zeros(walls.shape)
    for row in range(walls.shape[0]):
        for col in range(walls.shape[1]):
            if walls[row, col]:
                continue
            colors = {
                segments[n]
                for n in neighbors((row, col), walls.shape)
                if segments[n] != 0
            }
            if colors:
                c = min(colors)
                for other in colors:
                    if other == c:
                        continue
                    segments[np.where(segments == other)] = c
                segments[row, col] = c
            else:
                segments[row, col] = seg
                seg += 1
    return segments.astype(int)  # type: ignore


def generate_board(
    rows: int,
    cols: int,
    iterations: int = 4,
    hills_per_player: int = 2,
    hill_dist=0.3,
    starting_percent_open: float = 0.5,
    min_open: float = 0.4,
    max_open: float = 0.7,
    percent_food: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> Board:
    rng = rng if rng else np.random.default_rng()
    while True:
        walls = _erode_caverns(rows, cols, iterations, starting_percent_open, rng)
        seg = _segment(walls)
        counts = np.bincount(seg.ravel())
        biggest = np.argmax(counts[1:]) + 1
        walls[np.where(seg != biggest)] = 1
        p_open = 1 - (np.sum(walls)) / (rows * cols)
        if min_open < p_open < max_open:
            break
    hills = _spawn_hills(rows, cols, hills_per_player, hill_dist, rng, walls)

    out = Board(walls, hills)
    while np.count_nonzero(out.food) < percent_food * np.count_nonzero(walls):
        out.spawn_food()
    return out


def _spawn_hills(
    rows: int,
    cols: int,
    hills_per_ployer: int,
    hill_dist: float,
    rng: np.random.Generator,
    walls: npt.NDArray[np.int_],
):
    min_dist_2 = (hill_dist * max(rows, cols)) ** 2
    while True:
        hills = np.zeros((rows, cols)).astype(int)
        open_cells = list(zip(*np.where(walls == 0)))
        p_hills = rng.choice(open_cells, hills_per_ployer, False)
        for hill in p_hills:
            r, c = hill
            hills[r, c] = 1
        p_hills2 = np.flipud(np.fliplr(hills))
        hills[np.where(p_hills2 == 1)] = 2
        hill_locations = zip(*np.where(hills != 0))
        good = True
        for a, b in combinations(hill_locations, 2):
            d2 = toroidal_distance_2(a, b, (rows, cols))
            if d2 < min_dist_2:
                good = False
                break
        if good:
            return hills


def _erode_caverns(rows, cols, iterations, starting_percent_open, rng):
    walls = (rng.uniform(0, 1, (rows, cols)) < starting_percent_open).astype(int)
    walls[:, cols // 2 :] = np.flipud(np.fliplr(walls[:, : cols // 2]))
    for _ in range(iterations):
        wall_neighbors = convolve2d(
            walls, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), boundary="wrap"
        )[1:-1, 1:-1]
        next_walls = np.zeros((rows, cols))
        next_walls[np.where(walls)] = (wall_neighbors >= 4)[np.where(walls)]
        next_walls[np.where(1 - walls)] = (wall_neighbors >= 5)[np.where(1 - walls)]
        walls = next_walls.astype(int)
    return walls


def main():
    fps = 60
    fps_clock = pygame.time.Clock()
    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))

    game_board = generate_board(
        100, 100, iterations=6, hills_per_player=3, min_open=0.45
    )
    drawing = game_board.render(screen.get_width(), screen.get_height())

    while True:
        screen.fill("#ffffff")

        for event in pygame.event.get():
            if event.type == pygame.locals.QUIT:
                pygame.quit()
                sys.exit()

        screen.blit(drawing, (0, 0))

        pygame.display.flip()
        fps_clock.tick(fps)


if __name__ == "__main__":
    main()
