"""Class representing a group of particles moving in a plasma's fields."""
import numpy as np
from astropy import units as u
import tqdm.auto
import xarray
import warnings
import matplotlib.pyplot as plt

from plasmapy import atomic, formulary
from plasmapy.utils.decorators import check_units
from plasmapy.utils import PhysicsError
from plasmapy.atomic import particle_input, Particle
import typing
from . import particle_integrators

PLOTTING = False

__all__ = ["ParticleTracker", "ParticleTrackerAccessor"]


@check_units
@particle_input
def _create_xarray(
    position_history: u.m,
    velocity_history: u.m / u.s,
    times: u.s,
    b_history: u.T,
    e_history: u.V / u.m,
    timesteps: u.s,
    particle: Particle,
    dimensions="xyz",
):
    data_vars = {}
    assert position_history.shape == velocity_history.shape
    particles = range(position_history.shape[1])
    data_vars["position"] = (
        ("time", "particle", "dimension"),
        position_history.si.value,
    )
    data_vars["velocity"] = (
        ("time", "particle", "dimension"),
        velocity_history.si.value,
    )
    data_vars["B"] = (("time", "particle", "dimension"), b_history.si.value)
    data_vars["E"] = (("time", "particle", "dimension"), e_history.si.value)
    data_vars["timestep"] = (("time",), timesteps.si.value)

    data = xarray.Dataset(
        data_vars=data_vars,
        coords={
            "time": times.si.value,
            "particle": particles,
            "dimension": list(dimensions),
        },
    )
    for index, quantity in [
        ("position", position_history),
        ("velocity", velocity_history),
        ("time", times),
        ("B", b_history),
        ("E", e_history),
        ("timestep", timesteps),
    ]:
        data[index].attrs["unit"] = str(quantity.unit)

    data.attrs["particle"] = str(particle)
    return data


@xarray.register_dataset_accessor("particletracker")
class ParticleTrackerAccessor:
    def __init__(self, xarray_obj, plasma=None):
        self._obj = xarray_obj
        self.plasma = plasma
        self.particle = Particle(xarray_obj.attrs["particle"])
        # self.diagnostics = diagnostics # TODO put in xarray itself

    def plot_trajectories(self, *args, **kwargs):  # coverage: ignore
        r"""Draws trajectory history."""
        from astropy.visualization import quantity_support
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        quantity_support()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for p_index in range(self.particle.size):
            r = self.position.isel(particle=p_index)
            x, y, z = r.T
            ax.plot(x, y, z, *args, **kwargs)
        ax.set_xlabel("$x$ position")
        ax.set_ylabel("$y$ position")
        ax.set_zlabel("$z$ position")
        plt.show()

    def plot_time_trajectories(self, plot="xyz"):  # coverage: ignore
        r"""
        Draws position history versus time.

        Parameters
        ----------
        plot : str (optional)
            Enable plotting of position component x, y, z for each of these
            letters included in `plot`.
        """
        from astropy.visualization import quantity_support
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        quantity_support()
        fig, ax = plt.subplots()
        for p_index in range(self.particle.size):
            r = self.position.isel(particle=p_index)
            x, y, z = r.T
            if "x" in plot:
                ax.plot(self.time, x, label=f"x_{p_index}")
            if "y" in plot:
                ax.plot(self.time, y, label=f"y_{p_index}")
            if "z" in plot:
                ax.plot(self.time, z, label=f"z_{p_index}")
        # ax.set_title(self.name)
        ax.legend(loc="best")
        ax.grid()
        ax.set_xlabel(f"Time $t$ [{u.s}]")
        ax.set_ylabel(f"Position [{u.m}]")
        plt.show()

    def test_kinetic_energy(self, cutoff=2):
        r"""Test conservation of kinetic energy."""
        difference = self.kinetic_energy - self.kinetic_energy.mean(dim="time")
        scaled = difference / self.kinetic_energy.std(dim="time")
        conservation = abs(scaled) < cutoff
        if not conservation.all():
            if PLOTTING:
                import matplotlib.pyplot as plt

                self.kinetic_energy.plot.line()
                plt.show()
            raise PhysicsError("Kinetic energy is not conserved!")

    def visualize(self, figure=None, particle=0):  # coverage: ignore
        """Plot the trajectory using PyVista."""
        # breakpoint()
        import pyvista as pv

        if figure is None:
            fig = pv.Plotter()
            fig.add_axes()
        else:
            fig = figure
        points = self.position.sel(particle=particle).values
        # breakpoint()
        trajectory = spline = pv.Spline(points, 1000)
        if self.plasma and hasattr(self.plasma, "visualize"):
            self.plasma.visualize(fig)

        if figure is None:
            trajectory.plot(fig)
            fig.show()
        else:
            fig.add_mesh(trajectory)
        return fig

    @property
    def kinetic_energy(self):
        r"""
        Calculate the kinetic energy history for each particle.

        Returns
        --------
        ~astropy.units.Quantity
            Array of kinetic energies, shape (nt, n).
        """
        if "kinetic_energy" not in self._obj:
            kinetic_energy = (
                (self._obj.velocity ** 2).sum(axis=-1) * self.particle.mass / 2
            )
            self._obj["kinetic_energy"] = (("time", "particle"), kinetic_energy)
            self._obj["kinetic_energy"].attrs["unit"] = "J"
        return self._obj.kinetic_energy


class ParticleTracker:
    """
    A group of particles moving in a plasma's electromagnetic field.

    Parameters
    ----------
    plasma : `Plasma`
        plasma from which fields can be pulled
    type : str
        particle type. See `plasmapy.atomic.atomic` for suitable arguments.
        The default is a proton.
    x: `astropy.units.Quantity`
    v: `astropy.units.Quantity`
        Initial conditions for position and velocity, with a shape of (N, 3),
        N being the number of particles in your simulation.

    Attributes
    ----------
    _x : `np.ndarray`
    _v : `np.ndarray`
        Current position and velocity, without units. Shape (N, 3).
    q : `astropy.units.Quantity`
    m : `astropy.units.Quantity`
        Charge and mass of particle.

    Examples
    ----------
    See `plasmapy/examples/plot_particle_stepper.ipynb`.

    """

    integrators = {
        "explicit_boris": particle_integrators._boris_push,
        "implicit_boris": particle_integrators._boris_push_implicit,
        "implicit_boris2": particle_integrators._boris_push_implicit2,
        "zenitani": particle_integrators._zenitani,
    }

    @atomic.particle_input
    @check_units()
    def __init__(
        self,
        plasma,
        x: u.m = None,
        v: u.m / u.s = None,
        particle_type: atomic.Particle = "p",
    ):

        if x is not None and v is not None:
            pass
        elif x is None and v is None:
            x = u.Quantity(np.zeros((1, 3)), u.m)
            v = u.Quantity(np.zeros((1, 3)), u.m / u.s)
        elif v is not None:
            x = u.Quantity(np.zeros((v.shape)), u.m)
        elif x is not None:
            v = u.Quantity(np.zeros((x.shape)), u.m / u.s)
        self.q = particle_type.charge
        self.m = particle_type.mass
        self._m = particle_type.mass.si.value
        self.particle = particle_type
        self.name = particle_type.particle

        self.plasma = plasma

        self._x = x.si.value
        self._v = v.si.value
        assert v.shape == x.shape
        self.N, dims = x.shape
        assert dims == 3

        self._check_field_size()

    def _check_field_size(self):
        b = self.plasma.interpolate_B(self.x)
        e = self.plasma.interpolate_E(self.x)
        if b.shape != self._x.shape:
            raise ValueError(
                f"""Invalid shape {b.shape} for the magnetic field array!
                `plasma.interpolate_B` must return an array of shape (N, 3),
                where N is the number of particles in the simulation, currently {self.N}."""
            )
        if e.shape != self._x.shape:
            raise ValueError(
                f"""Invalid shape {e.shape} for the electric field array!
                `plasma.interpolate_E` must return an array of shape (N, 3),
                where N is the number of particles in the simulation, currently {self.N}."""
            )

    @property
    def x(self):
        """Particle position (as Astropy Quantity)."""
        return u.Quantity(self._x, u.m, copy=False)

    # @check_units() # TODO
    @x.setter
    def x(self, value: u.m):
        self._x = value.si.value

    @property
    def v(self):
        """Particle velocity (as Astropy Quantity)."""
        return u.Quantity(self._v, u.m / u.s, copy=False)

    # @check_units()
    @v.setter
    def v(self, value: u.m / u.s):
        self._v = value.si.value

    @check_units()
    def run(
        self,
        total_time: u.s,
        dt: u.s = None,
        progressbar=True,
        pusher="explicit_boris",
        progressbar_steps=100,
    ):
        r"""Run a simulation instance.

        dt: u.s = np.inf * u.s,
        nt: int = np.inf,
        """
        integrator = self.integrators[pusher]
        if dt is None:
            b = np.linalg.norm(self.plasma.interpolate_B(self.x), axis=-1)
            gyroperiod = (1 / formulary.gyrofrequency(b, self.particle, to_hz=True)).to(
                u.s
            )
            dt = gyroperiod.min() / 20
            warnings.warn(
                f"Set timestep to {dt:.3e}, 1/20 of smallest gyroperiod", UserWarning
            )

        _dt = dt.si.value
        _q = self.particle.charge.si.value
        _m = self.particle.mass.si.value

        _total_time = total_time.si.value
        _time = 0.0
        _progressbar_timestep = _total_time / progressbar_steps
        next_progressbar_update_time = _time + _progressbar_timestep
        _times = [_time]
        _timesteps = [_dt]
        _x = self._x.copy()
        _v = self._v.copy()

        init_kinetic = self._kinetic_energy(_v)
        timestep_info = dict(i=len(_times), dt=_dt)
        if init_kinetic:
            reldelta = self._kinetic_energy(_v) / init_kinetic - 1
            kinetic_info = {"Relative kinetic energy change": reldelta}
        else:
            delta = self._kinetic_energy(_v)
            kinetic_info = {"Kinetic energy change": delta}

        with np.errstate(all="raise"):
            b = self.plasma._interpolate_B(_x)
            e = self.plasma._interpolate_E(_x)

            integrator(
                _x.copy(), _v, b, e, _q, _m, -0.5 * _dt
            )  # we don't want to change position here

            _position_history = [_x.copy()]
            _velocity_history = [_v.copy()]
            _b_history = [b.copy()]
            _e_history = [e.copy()]
            if progressbar:
                pbar = tqdm.auto.tqdm(total=progressbar_steps)
            while _time < _total_time:
                _time += _dt
                b = self.plasma._interpolate_B(_x)
                e = self.plasma._interpolate_E(_x)
                integrator(_x, _v, b, e, _q, _m, _dt)

                # todo should be a list of dicts, probably)
                _position_history.append(_x.copy())
                _velocity_history.append(_v.copy())
                _b_history.append(b.copy())
                _e_history.append(e.copy())
                _times.append(_time)
                _timesteps.append(_dt)

                if progressbar and _time > next_progressbar_update_time:
                    next_progressbar_update_time += _progressbar_timestep
                    diagnostics = dict(i=len(_times), dt=_dt)
                    if init_kinetic:
                        reldelta = self._kinetic_energy(_v) / init_kinetic - 1
                        diagnostics["Relative kinetic energy change"] = reldelta
                    else:
                        delta = self._kinetic_energy(_v)
                        diagnostics["Kinetic energy change"] = delta
                    pbar.set_postfix(diagnostics)
                    pbar.update()
        if progressbar:
            pbar.close()

        solution = _create_xarray(
            u.Quantity(_position_history, u.m),
            u.Quantity(_velocity_history, u.m / u.s),
            u.Quantity(_times, u.s),
            u.Quantity(_b_history, u.T),
            u.Quantity(_e_history, u.V / u.m),
            u.Quantity(_timesteps, u.s),
            self.particle,
        )
        return solution

    def _kinetic_energy(self, _v=None):
        if _v is None:
            _v = self._v
        return (_v ** 2).sum() * self._m / 2

    def kinetic_energy(self, v=None):
        """Calculate particle kinetic energy."""
        if v is None:
            v = self.v
        return u.Quantity(self._kinetic_energy(v.si.value), u.J)

    def __repr__(self, *args, **kwargs):
        return (
            f"ParticleTracker(plasma={self.plasma}, particle_type={self.particle},"
            f"N = {self.x.shape[0]})"
        )

    def __str__(self):  # coverage: ignore
        return f"{self.N} {self.name} with " f"q = {self.q:.2e}, m = {self.m:.2e}"
