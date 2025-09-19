import abc

import dataclasses
from collections.abc import Callable, Mapping
from typing import List

# we use Type to distinguish from type attr of PerturbationSelector
from typing import Any, ClassVar, Type  # noqa: UP035

import os
import dacite
import polling2
import datetime
import numpy as np
import torch
import xarray as xr

from fme.core.registry.registry import Registry


@dataclasses.dataclass
class PerturbationConfig(abc.ABC):
    """
    Returns a perturbation function config class.
    """

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "PerturbationConfig":
        """
        Create a PerturbationSelector from a dictionary containing all the information
        needed to build a PerturbationConfig.
        """
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @abc.abstractmethod
    def apply_perturbation(
        self,
        data: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        time: List[datetime.datetime] | List[datetime.timedelta] = None
    ) -> None: ...


@dataclasses.dataclass
class PerturbationSelector:
    type: str
    config: Mapping[str, Any]
    registry: ClassVar[Registry[PerturbationConfig]] = Registry[PerturbationConfig]()

    def __post_init__(self):
        if not isinstance(self.registry, Registry):
            raise ValueError("PerturbationSelector.registry should not be set manually")

    @classmethod
    def register(
        cls, type_name
    ) -> Callable[[Type[PerturbationConfig]], Type[PerturbationConfig]]:  # noqa: UP006
        return cls.registry.register(type_name)

    def build(self) -> PerturbationConfig:
        return self.registry.get(self.type, self.config)

    @classmethod
    def get_available_types(cls):
        """This class method is used to expose all available types of Perturbations."""
        return cls(type="", config={}).registry._types.keys()


@dataclasses.dataclass
class SSTPerturbation:
    """
    Configuration for sea surface temperature perturbations
    applied to initial condition and forcing data.
    Currently, this is strictly applied to both.

    Parameters:
        sst: List of perturbation selectors for SST perturbations.
    """

    perturbation_list: list[PerturbationSelector]

    def __post_init__(self):
        self.perturbations: list[PerturbationConfig] = [
            perturbation.build() for perturbation in self.perturbation_list
        ]



def _get_mask(fraction: torch.Tensor, cutoff: float = 0.5) -> torch.Tensor:
    return fraction > cutoff


@PerturbationSelector.register("constant")
@dataclasses.dataclass
class ConstantConfig(PerturbationConfig):
    """
    Configuration for a constant perturbation.
    """

    parameter_name: str
    mask_fraction_name: str
    amplitude: float = 1.0

    def apply_perturbation(
        self,
        data: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        time: datetime.timedelta = None
    ):  

        mask = _get_mask(data[self.mask_fraction_name])
        data[self.parameter_name][mask] += self.amplitude  # type: ignore
        
@PerturbationSelector.register("multiply")
@dataclasses.dataclass
class MultiplyConfig(PerturbationConfig):
    """
    Configuration for a multiply perturbation.
    """

    parameter_name: str
    mask_fraction_name: str
    amplitude: float = 1.0

    def apply_perturbation(
        self,
        data: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        time: datetime.timedelta = None
    ):  

        mask = _get_mask(data[self.mask_fraction_name])
        data[self.parameter_name][mask] *= self.amplitude  # type: ignore
        
@PerturbationSelector.register("from_file")
@dataclasses.dataclass
class FromFileConfig(PerturbationConfig):
    """
    Configuration for a perturbation loaded from a file.
    """

    parameter_name: str
    data_directory: str
    file_prefix: str
    parameter_name_in_file: str
    file_mask_fraction_name: str
    polling_timeout: int = 600  # seconds
    file_suffix: str = ""

    def apply_perturbation(
        self,
        data: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        time: List[datetime.timedelta] # We have to use time deltas since the ocean model only has access to timedeltas
    ):
        # Time contains the initialisation time, and the target time
        assert len(time) == 2, "FromFileConfig only supports one forward step in memory"
        
        init_dt = time[0]
        
        # Note; this requires the assumption that the variable here is slowly varying compared ot the atmosphere, (e.g. ocean, ice)
        # as we make the assumption that we can use the value at init time for the target time.
        ds_at_init_time = polling2.poll(lambda: xr.load_dataset(f"{self.data_directory}/{self.file_prefix}_{int((init_dt * 1e-9) / 3600)}h{self.file_suffix}.nc"),
                        ignore_exceptions=(IOError, ValueError, FileNotFoundError),
                        timeout=self.polling_timeout,
                        step=0.1).isel(time=0).transpose('latitude', 'longitude')
        
        # Create mask; this is because ACE forcing data seems to have non-zero sea ice fraction over land, and removing this may cause issues.
        mask_da = ~np.isnan(ds_at_init_time[self.file_mask_fraction_name])
        mask = torch.tensor(mask_da.values, device=data[self.parameter_name].device, dtype=torch.bool)
        mask = mask.expand(data[self.parameter_name].shape)
        
        replacement_tensor = torch.tensor(ds_at_init_time[self.parameter_name_in_file].fillna(0.0).values, device=data[self.parameter_name].device, dtype=data[self.parameter_name].dtype)
        replacement_tensor = replacement_tensor.expand(data[self.parameter_name].shape)
        
        data[self.parameter_name] = torch.where(mask, replacement_tensor, data[self.parameter_name])

@PerturbationSelector.register("greens_function")
@dataclasses.dataclass
class GreensFunctionConfig(PerturbationConfig):
    """
    Configuration for a single sinusoidal patch of a Green's function perturbation.
    See equation 1 in Bloch‐Johnson, J., et al. (2024).

    Parameters:
        amplitude: The amplitude of the perturbation,
            maximum is reached at (lat_center, lon_center).
        lat_center: The latitude at the center of the patch in degrees.
        lon_center: The longitude at the center of the patch in degrees.
        lat_width: latitudinal width of the patch in degrees.
        lon_width: longitudinal width of the patch in degrees.
    """
    mask_fraction_name: str
    amplitude: float = 1.0
    lat_center: float = 0.0
    lon_center: float = 0.0
    lat_width: float = 10.0
    lon_width: float = 10.0

    def __post_init__(self):
        self._lat_center_rad = np.deg2rad(self.lat_center)
        self._lon_center_rad = np.deg2rad(self.lon_center)
        self._lat_width_rad = np.deg2rad(self.lat_width)
        self._lon_width_rad = np.deg2rad(self.lon_width)

    def _wrap_longitude_discontinuity(
        self,
        lon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Assume longitude is in the range [0, 360) degrees.
        If the patch crosses the discontinuity at 0/360 degrees,
        shift the longitude accordingly.
        """
        lon_min = self.lon_center - self.lon_width / 2.0
        lon_max = self.lon_center + self.lon_width / 2.0
        if lon_min < 0:
            lon_shifted = ((lon + 180) % 360) - 180
            lon_in_patch = (lon_shifted > lon_min) & (lon_shifted < lon_max)
        elif lon_max > 360:
            lon_in_patch = (lon > lon_min) | (lon < lon_max % 360)
            lon_shifted = ((lon + 180) % 360) + 180
        else:
            lon_in_patch = (lon > lon_min) & (lon < lon_max)
            lon_shifted = lon
        return lon_in_patch, lon_shifted

    def apply_perturbation(
        self,
        data: torch.Tensor,
        lat: torch.Tensor,
        lon: torch.Tensor,
        time: datetime.timedelta = None
    ):
        lat_in_patch = torch.abs(lat - self.lat_center) < self.lat_width / 2.0
        lon_in_patch, lon_shifted = self._wrap_longitude_discontinuity(lon)
        mask = lat_in_patch & lon_in_patch
        ocean_mask = _get_mask(data[self.mask_fraction_name])
        perturbation = self.amplitude * (
            torch.cos(
                torch.pi
                / 2
                * (lat.deg2rad() - self._lat_center_rad)
                / (self._lat_width_rad / 2.0)
            )
            ** 2
            * torch.cos(
                torch.pi
                / 2
                * (lon_shifted.deg2rad() - self._lon_center_rad)
                / (self._lon_width_rad / 2.0)
            )
            ** 2
        )
        mask = mask.expand(data.shape)
        perturbation = perturbation.expand(data.shape)
        data[mask & ocean_mask] += perturbation[mask & ocean_mask]
