"""
mission_orientation.py

Lets every flight pattern be flown in any of the 8 orientations of the
rectangular mission area (the symmetries of a rectangle: which edge the swarm
starts from, which way it heads, and which side its lanes are ordered from).

Patterns are generated in a canonical frame and then mapped into the world
frame. For orientations that swap the axes, the canonical frame is the
mission area with width and height exchanged, so trajectories always stay
inside the real bounds.

Orientation index o = 4*swap + 2*flip_y + flip_x  (0..7)

    o  swap flip_y flip_x   canonical +x sweep becomes ...
    0   0     0      0      west  -> east   (the original, fixed orientation)
    1   0     0      1      east  -> west
    2   0     1      0      west  -> east, lanes ordered north -> south
    3   0     1      1      east  -> west, lanes ordered north -> south
    4   1     0      0      south -> north
    5   1     0      1      north -> south
    6   1     1      0      south -> north, lanes ordered east -> west
    7   1     1      1      north -> south, lanes ordered east -> west

Wind is defined in the world frame and transformed into the canonical frame,
so it blows the same geographic direction in every orientation.
"""

from dataclasses import replace
import numpy as np

N_ORIENTATIONS = 8
ORIENTATION_NAMES = {
    0: "W->E", 1: "E->W", 2: "W->E (lanes N->S)", 3: "E->W (lanes N->S)",
    4: "S->N", 5: "N->S", 6: "S->N (lanes E->W)", 7: "N->S (lanes E->W)",
}


def _flags(orientation: int):
    if not 0 <= int(orientation) < N_ORIENTATIONS:
        raise ValueError(f"orientation must be 0..{N_ORIENTATIONS - 1}")
    o = int(orientation)
    return bool(o & 4), bool(o & 2), bool(o & 1)   # swap, flip_y, flip_x


def canonical_bounds(world_bounds, orientation: int):
    """Bounds of the frame in which the pattern is generated."""
    swap, _, _ = _flags(orientation)
    if not swap:
        return world_bounds
    return replace(world_bounds,
                   x_min=world_bounds.y_min, x_max=world_bounds.y_max,
                   y_min=world_bounds.x_min, y_max=world_bounds.x_max)


def world_to_canonical_vector(vec: np.ndarray, orientation: int) -> np.ndarray:
    """Transform a world-frame direction vector (e.g. wind) to canonical."""
    swap, flip_y, flip_x = _flags(orientation)
    v = np.array(vec, dtype=float).copy()
    # world = S(F(canonical)) where F flips then S swaps; invert: unswap, unflip
    if swap:
        v[0], v[1] = v[1], v[0]
    if flip_x:
        v[0] = -v[0]
    if flip_y:
        v[1] = -v[1]
    return v


def canonical_to_world(traj: np.ndarray, world_bounds, orientation: int) -> np.ndarray:
    """
    Map trajectories [..., 3] from the canonical frame to the world frame.
    z is unchanged.
    """
    swap, flip_y, flip_x = _flags(orientation)
    cb = canonical_bounds(world_bounds, orientation)
    out = np.array(traj, dtype=float, copy=True)
    x = out[..., 0].copy(); y = out[..., 1].copy()
    if flip_x:
        x = cb.x_min + cb.x_max - x
    if flip_y:
        y = cb.y_min + cb.y_max - y
    if swap:
        x, y = y, x
    out[..., 0] = x; out[..., 1] = y
    return out


def generate_oriented_swarm(generator_cls, world_bounds, orientation: int, **gen_kwargs):
    """
    Build a generator in the canonical frame for this orientation, with wind
    expressed in that frame. Returns (generator, to_world) where to_world maps
    the generator's trajectories into the world frame.
    """
    gen = generator_cls(canonical_bounds(world_bounds, orientation), **gen_kwargs)
    gen.wind_velocity = world_to_canonical_vector(gen.wind_velocity, orientation)

    def to_world(traj):
        return canonical_to_world(traj, world_bounds, orientation)

    return gen, to_world
