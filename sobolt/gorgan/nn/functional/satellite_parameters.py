from typing import Dict


class SatParams:
    """A class containing satellite parameters used by the physical downsampling
    algorithm.

    It has convenience methods allowing users to translate between high / low resolution
    satellite parameters. For example:

        real_s2 = SatParams.from_config(s2_config)
        high_s2 = real_s2 * 4  # Sat params for a satellite capturing higher res imagery
        low_s2 = real_s2 / 4   # Sat params for a satellite capturing lower res imagery
    """

    altitude: float
    aperture_x: float
    aperture_y: float
    focal_distance: float
    resolution: float

    wavelength_blue: float
    wavelength_green: float
    wavelength_red: float
    wavelength_infrared: float

    @staticmethod
    def from_config(config: Dict[str, float]) -> "SatParams":
        """Instantiate the class based on a dictionary."""
        return SatParams(
            config["altitude"],
            config["aperture_x"],
            config["aperture_y"],
            config["focal_distance"],
            config["resolution"],
            config["wavelength_blue"],
            config["wavelength_green"],
            config["wavelength_red"],
            config["wavelength_infrared"],
        )

    def __init__(
        self,
        altitude: float,
        aperture_x: float,
        aperture_y: float,
        focal_distance: float,
        resolution: float,
        wavelength_blue: float,
        wavelength_green: float,
        wavelength_red: float,
        wavelength_infrared: float,
    ):
        self.altitude = altitude
        self.aperture_x = aperture_x
        self.aperture_y = aperture_y
        self.focal_distance = focal_distance
        self.resolution = resolution
        self.wavelength_blue = wavelength_blue
        self.wavelength_green = wavelength_green
        self.wavelength_red = wavelength_red
        self.wavelength_infrared = wavelength_infrared

    def __mul__(self, factor: float) -> "SatParams":
        """Create a set of satellite parameters based on the current set of parameters.

        Values greater than one lead to satellite parameters for *higher* resolution
        imagery. Values between 0 and 1 lead to *lower* resolution satellite parameters.

        To do so, the aperture sizes are divided by the given factor. In other words, if
        the current instance has Sentinel-2 parameters and the given factor is 4, the
        result is the parameters for a Sentinel-2 satellite with 4 times higher
        resolution.

            new_aperture_x = aperture_x / factor = 150e-3 / 4
            new_aperture_y = aperture_y / factor = 150e-3 / 4
            new_resolution = resolution / factor = 10.0 / 4

        The values for wavelengths, altitude and focal distance are left untouched.

        Parameters
        ----------
        divider: float
            The number to divide the aperture and resolution with.
        """
        return SatParams(
            self.altitude,
            self.aperture_x / factor,
            self.aperture_y / factor,
            self.focal_distance,
            self.resolution / factor,
            self.wavelength_blue,
            self.wavelength_green,
            self.wavelength_red,
            self.wavelength_infrared,
        )

    def __truediv__(self, factor) -> "SatParams":
        """Create a set of satellite parameters based on the current set of parameters.

        Values greater than one lead to satellite parameters for *lower* resolution
        imagery. Values between 0 and 1 lead to *higher* resolution satellite parameters.
        """
        return self * (1 / factor)


class SatParamsSentinel2(SatParams):
    """Sentinel-2 satellite parameters."""

    def __init__(self):
        super().__init__(
            786e3, 150e-3, 150e-3, 600e-3, 10.0, 492.4e-9, 559.8e-9, 664.6e-9, 832.8e-9
        )


class SatParamsSuperView(SatParams):
    """SuperView satellite parameters."""

    def __init__(self):
        super().__init__(
            530e3, 625e-3, 625e-3, 10000e-3, 2.0, 485e-9, 555e-9, 660e-9, 830e-9
        )
