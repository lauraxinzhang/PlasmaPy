from plasmapy.classes.plasma_base import GenericPlasma
import astropy.units as u
import numpy as np

E_unit = u.N

import math
import numba


@numba.njit
def scalar_distance(r, distances, directions):
    number_particles, dimensionality = r.shape
    for particle_i in range(number_particles):
        for particle_j in range(number_particles):
            scalar_distance = 0
            for dimension in range(dimensionality):
                diff = r[particle_i, dimension] - r[particle_j, dimension]
                directions[particle_i, particle_j, dimension] = diff
                scalar_distance += diff ** 2
            scalar_distance = math.sqrt(scalar_distance)
            distances[particle_i, particle_j] = scalar_distance
            if scalar_distance > 0:
                for dimension in range(dimensionality):
                    directions[particle_i, particle_j, dimension] /= scalar_distance


def get_forces_python(r, forces, potentials, A, B):
    for particle_i in numba.prange(r.shape[0]):
        force_on_i = np.zeros(r.shape[1])
        potential_on_i = 0.0
        for particle_j in range(r.shape[0]):
            if particle_i != particle_j:
                square_distance = np.sum((r[particle_i] - r[particle_j]) ** 2)
                assert square_distance > 0
                repulsive_part = B * square_distance ** -3
                attractive_part = A * square_distance ** -6

                if forces is not None:
                    force_term = 2 * attractive_part - repulsive_part
                    force_on_i += (
                        (r[particle_i] - r[particle_j]) / square_distance * force_term
                    )
                if potentials is not None:
                    potential_on_i += 2 * (attractive_part - repulsive_part)

        if forces is not None:
            forces[particle_i] = 24 * force_on_i
        if potentials is not None:
            potentials[particle_i] = potential_on_i


def add_wall_forces_python(r, forces, potentials, L, constant, scale, exponent=2):
    for particle_i in numba.prange(len(r)):
        R = r[particle_i]
        for i in range(3):
            if R[i] < 0:
                scaled_distance = R[i] / scale
                forces[particle_i, i] -= constant * scaled_distance ** exponent
                potentials[particle_i] += (
                    constant * scaled_distance ** (exponent + 1) / exponent
                )
            elif R[i] > L:
                scaled_distance = (R[i] - L) / scale
                forces[particle_i, i] -= constant * scaled_distance ** exponent
                potentials[particle_i] += (
                    constant * scaled_distance ** (exponent + 1) / exponent
                )


get_forces_njit = numba.njit()(get_forces_python)
get_forces_njit_parallel = numba.njit(parallel=True)(get_forces_python)

add_wall_forces_njit = numba.njit()(add_wall_forces_python)
add_wall_forces_njit_parallel = numba.njit(parallel=True)(add_wall_forces_python)

calculators = {
    "python": get_forces_python,
    "njit": get_forces_njit,
    "njit_parallel": get_forces_njit_parallel,
}

wall_forces = {
    "python": add_wall_forces_python,
    "njit": add_wall_forces_njit,
    "njit_parallel": add_wall_forces_njit_parallel,
}


def wall_collision(r, v, box_L):
    number_particles, dimensionality = r.shape
    for particle_i in numba.prange(number_particles):
        x, y, z = r[particle_i]
        if not (0 < x < box_L):
            v[particle_i, 0] *= -1
        if not (0 < y < box_L):
            v[particle_i, 1] *= -1
        if not (0 < z < box_L):
            v[particle_i, 2] *= -1


wall_collision_njit = numba.njit()(wall_collision)
wall_collision_njit_parallel = numba.njit(parallel=True)(wall_collision)

wall_collisions = {
    "python": wall_collision,
    "njit": wall_collision_njit,
    "njit_parallel": wall_collision_njit_parallel,
}


class InterParticleForces(GenericPlasma):
    def __init__(
        self,
        mechanism: ["python", "njit", "njit_parallel", "njit_cuda"],
        potential_well_depth: float,
        zero_distance: float,
        box_L: float,
        box_constant: float,
        box_force_scale: float,
        box_exponent: float = 2,
    ):
        self.potential_well_depth = potential_well_depth
        self.zero_distance = zero_distance
        self.mechanism = mechanism
        self.A = 4 * potential_well_depth * zero_distance ** 12
        self.B = 4 * potential_well_depth * zero_distance ** 6
        self.box_L = box_L
        self.box_constant = box_constant
        self.box_force_scale = box_force_scale
        self.box_exponent = box_exponent
        self.forces = None
        self.potentials = None

    def _interpolate_E(self, r: np.ndarray):
        number_particles, dimensionality = r.shape
        if self.forces is None:
            self.forces = np.zeros_like(
                r
            )  # TODO handle dtype - should be preallocated anyway for cuda so it's fine
        # else:
        #     self.forces[...] = 0.0

        if self.potentials is None:
            self.potentials = np.zeros(number_particles, dtype=float)
        # else:
        #     self.potentials[...] = 0.0

        calculators[self.mechanism](r, self.forces, self.potentials, self.A, self.B)
        wall_forces[self.mechanism](
            r,
            self.forces,
            self.potentials,
            self.box_L,
            self.box_constant,
            self.box_force_scale,
            self.box_exponent,
        )
        return self.forces

    def interpolate_E(self, r: u.m):
        return u.Quantity(self._interpolate_E(r.si.value), E_unit)

    def _interpolate_B(self, r):
        return np.zeros_like(r)

    def interpolate_B(self, r: u.m):
        return u.Quantity(self._interpolate_B(r.si.value), u.T)

    def _drag(self, r: np.ndarray, v: np.ndarray):
        # wall_collisions[self.mechanism](r, v, self.box_L)
        pass

    def visualize(self, figure=None):  # coverage: ignore
        import pyvista as pv

        if figure is None:
            fig = pv.Plotter(notebook=True)
        else:
            fig = figure

        L = self.box_L
        fig.add_mesh(pv.Cube(bounds=(0, L, 0, L, 0, L)), opacity=0.1)

        if figure is None:
            fig.show()

        return fig
