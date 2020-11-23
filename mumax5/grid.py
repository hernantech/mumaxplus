"""Grid implementation."""

import _mumax5cpp as _cpp


class Grid:
    """Create a Grid instance.

    Parameters
    ----------
    size : tuple[int] of size 3
        The grid size.
    origin : tuple[int] of size 3, optional
        The origin of the grid. The default value is (0, 0, 0).
    """

    def __init__(self, size, origin=(0, 0, 0)):
        self._impl = _cpp.Grid(size, origin)

    @classmethod
    def _from_impl(cls, impl):
        grid = cls.__new__(cls)
        grid._impl = impl
        return grid

    @property
    def size(self):
        """Return the current size."""
        return self._impl.size

    @property
    def origin(self):
        """Return the current origin."""
        return self._impl.origin

    @property
    def shape(self):
        """Return the current size."""
        Nx, Ny, Nz = self.size
        return (Nz, Ny, Nx)

    @property
    def ncells(self):
        """Return the total number of cells in this grid."""
        Nx, Ny, Nz = self.size
        return Nx * Ny * Nz
