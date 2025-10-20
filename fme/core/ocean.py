
import os
import pickle
import numpy as np
import dataclasses
import datetime
import polling2
import torch
import xarray as xr
from fme.core.typing_ import TensorDict, TensorMapping

from .atmosphere_data import AtmosphereData
from .constants import DENSITY_OF_WATER, SPECIFIC_HEAT_OF_WATER
from .prescriber import Prescriber


@dataclasses.dataclass
class SlabOceanConfig:
    """
    Configuration for a slab ocean model.

    Parameters:
        mixed_layer_depth_name: Name of the mixed layer depth field.
        q_flux_name: Name of the heat flux field.
    """

    mixed_layer_depth_name: str
    q_flux_name: str

    @property
    def names(self) -> list[str]:
        return [self.mixed_layer_depth_name, self.q_flux_name]

@dataclasses.dataclass
class FromFileOceanConfig:
    """
    Configuration for a ocean model where the ocean state is read from files (current workaround
    for a CPU-intensive dynamical ocean).

    Parameters:
        router_folder: Folder containing the router files.
        atmosphere_timestep_hours: Timestep of the atmosphere model in hours.
    """

    router_folder: str
    file_prefix: str
    grid_file_path: str
    polling_timeout: int = 60*10  # 10 minutes
    sea_ice_fraction_name: str = None
    file_suffix: str = ""
    

@dataclasses.dataclass
class OceanConfig:
    """
    Configuration for determining sea surface temperature from an ocean model.

    Parameters:
        surface_temperature_name: Name of the sea surface temperature field.
        ocean_fraction_name: Name of the ocean fraction field.
        interpolate: If True, interpolate between ML-predicted surface temperature and
            ocean-predicted surface temperature according to ocean_fraction. If False,
            only use ocean-predicted surface temperature where ocean_fraction>=0.5.
        slab: If provided, use a slab ocean model to predict surface temperature.
    """

    surface_temperature_name: str
    ocean_fraction_name: str
    interpolate: bool = False
    slab: SlabOceanConfig | None = None
    from_file: FromFileOceanConfig | None = None

    def build(
        self,
        in_names: list[str],
        out_names: list[str],
        timestep: datetime.timedelta,
    ) -> "Ocean":
        if not (
            self.surface_temperature_name in in_names
            and self.surface_temperature_name in out_names
        ):
            raise ValueError(
                "To use a surface ocean model, the surface temperature must be present"
                f" in_names and out_names, but {self.surface_temperature_name} is not."
            )
        return Ocean(config=self, timestep=timestep)

    @property
    def forcing_names(self) -> list[str]:
        names = [self.ocean_fraction_name]
        if self.slab is None:
            names.append(self.surface_temperature_name)
        else:
            names.extend(self.slab.names)
        return list(set(names))


class Ocean:
    """Overwrite sea surface temperature with that predicted from some ocean model."""

    def __init__(self, config: OceanConfig, timestep: datetime.timedelta):
        """
        Args:
            config: Configuration for the surface ocean model.
            timestep: Timestep of the model.
        """
        self.surface_temperature_name = config.surface_temperature_name
        self.ocean_fraction_name = config.ocean_fraction_name

        self.prescriber = Prescriber(
            prescribed_name=config.surface_temperature_name,
            mask_name=config.ocean_fraction_name,
            mask_value=1,
            interpolate=config.interpolate,
        )
        self._forcing_names = config.forcing_names
        if config.slab is None and config.from_file is None:
            self.type = "prescribed"
        elif config.slab is not None:
            self.type = "slab"
            self.mixed_layer_depth_name = config.slab.mixed_layer_depth_name
            self.q_flux_name = config.slab.q_flux_name
        elif config.from_file is not None:
            self.type = "from_file"
            self.router_folder = config.from_file.router_folder
            self.timestep_counter = 0
            self.polling_timeout = config.from_file.polling_timeout
            self.sea_ice_fraction_name = config.from_file.sea_ice_fraction_name
            self.file_prefix = config.from_file.file_prefix
            self.file_suffix = config.from_file.file_suffix
            self.grid_da = xr.load_dataarray(config.from_file.grid_file_path)
            
        self.timestep = timestep
        self.timestep_hrs = int(self.timestep.seconds / 3600)

    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
        target_data: TensorMapping,
    ) -> TensorDict:
        """
        Args:
            input_data: Denormalized input data for current step.
            gen_data: Denormalized output data for current step.
            target_data: Denormalized data that includes mask and forcing data. Assumed
                to correspond to the same time step as gen_data.

        Returns:
            gen_data with sea surface temperature overwritten by ocean model.
        """
        if self.type == "prescribed":
            next_step_temperature = target_data[self.surface_temperature_name]
            prescriber_dict = {self.surface_temperature_name: next_step_temperature}
        elif self.type == "slab":
            temperature_tendency = mixed_layer_temperature_tendency(
                AtmosphereData(gen_data).net_surface_energy_flux_without_frozen_precip,
                target_data[self.q_flux_name],
                target_data[self.mixed_layer_depth_name],
            )
            next_step_temperature = (
                input_data[self.surface_temperature_name]
                + temperature_tendency * self.timestep.total_seconds()
            )
            
            prescriber_dict = {self.surface_temperature_name: next_step_temperature}
            
        elif self.type == "from_file":

            # Write generated data to file for the router to read

            flux_dict = {k: (['latitude', 'longitude'], v.squeeze().cpu()) for k, v in gen_data.items()}
            ds = xr.Dataset(flux_dict, coords={'latitude': self.grid_da['latitude'].values, 'longitude': self.grid_da['longitude'].values})
            ds.to_netcdf(os.path.join(self.router_folder, f"ace2_{(self.timestep_counter + 1) * self.timestep_hrs}h.nc"),)

            # Load ocean data. Note, this must be on a 180 x 360 grid.
            ocean_ds = polling2.poll(lambda: xr.load_dataset(os.path.join(self.router_folder, f"{self.file_prefix}_{(self.timestep_counter + 1) * self.timestep_hrs}h{self.file_suffix}.nc")),
                    ignore_exceptions=(IOError, ValueError, FileNotFoundError),
                    timeout=self.polling_timeout,
                    step=0.1).isel(time=0).transpose('latitude', 'longitude')

            self.timestep_counter += 1

            # Make sure all SSTs under sea ice are set to just above freezing point of salt water, as done by ERA5
            # Note that, since incoming SST has null values at land points, we are implicitly masking out the land
            # Which is good because the ACE ocean fraction seems to be > 0 over some land points
            ice_frac = ocean_ds['sea_ice_fraction'].fillna(0.0)
            sst_da = (1 - ice_frac) * ocean_ds['sea_surface_temperature'] + ice_frac * ocean_ds['sea_ice_temperature']
            
            device = gen_data[self.surface_temperature_name].device
            sst_array = torch.tensor(sst_da.values, dtype=torch.float32).to(device)
            sst_array = sst_array[None, :, :] # Add batch dimension
            
            # Make sure there aren't any null values in the SST (will be interpolated )
            next_step_temperature = torch.where(torch.isnan(sst_array), gen_data[self.surface_temperature_name], sst_array)
            
            # Prescribe SST
            prescriber_dict = {self.surface_temperature_name: next_step_temperature}
                
        else:
            raise NotImplementedError(f"Ocean type={self.type} is not implemented")

        return self.prescriber(
            target_data,
            gen_data,
            prescriber_dict,
        )

    @property
    def forcing_names(self) -> list[str]:
        """These are the variables required from the forcing data."""
        return self._forcing_names


def mixed_layer_temperature_tendency(
    f_net: torch.Tensor,
    q_flux: torch.Tensor,
    depth: torch.Tensor,
    density=DENSITY_OF_WATER,
    specific_heat=SPECIFIC_HEAT_OF_WATER,
) -> torch.Tensor:
    """
    Args:
        f_net: Net surface energy flux in W/m^2.
        q_flux: Convergence of ocean heat transport in W/m^2.
        depth: Mixed layer depth in m.
        density (optional): Density of water in kg/m^3.
        specific_heat (optional): Specific heat of water in J/kg/K.

    Returns:
        Temperature tendency of mixed layer in K/s.
    """
    return (f_net + q_flux) / (density * depth * specific_heat)
