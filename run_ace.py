import os
import sys
import copy
import dataclasses
import datetime
import logging
from argparse import ArgumentParser

from collections.abc import Mapping, Sequence
from typing import Literal

import dacite
import pickle
import numpy as np
import torch
import xarray as xr
from xarray.coding.times import CFDatetimeCoder

sys.path.append('/home/a/antonio/repos/ace')

import fme
import fme.core.logging_utils as logging_utils
from fme.ace.aggregator.inference import InferenceAggregatorConfig
from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.data_loading.getters import get_forcing_data
from fme.ace.data_loading.perturbation import PerturbationSelector
from fme.ace.data_loading.inference import (
    ExplicitIndices,
    ForcingDataLoaderConfig,
    InferenceInitialConditionIndices,
    TimestampList,
)
from fme.ace.inference.data_writer import DataWriterConfig, DataWriter, PairedDataWriter
from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.stepper import (
    Stepper,
    StepperOverrideConfig,
    load_stepper,
    load_stepper_config,
)
from fme.ace.stepper.single_module import StepperConfig
from fme.core.cli import prepare_config, prepare_directory
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset_info import IncompatibleDatasetInfo
from fme.core.dicts import to_flat_dict
from fme.core.generics.inference import get_record_to_wandb, run_inference
from fme.core.logging_utils import LoggingConfig
from fme.core.timing import GlobalTimer

from fme.ace.inference.evaluator import resolve_variable_metadata, validate_time_coarsen_config

StartIndices = InferenceInitialConditionIndices | ExplicitIndices | TimestampList

@dataclasses.dataclass
class InitialConditionConfig:
    """
    Configuration for initial conditions.

    .. note::
        The data specified under path should contain a time dimension of at least
        length 1. If multiple times are present in the dataset specified by ``path``,
        the inference will start an ensemble simulation using each IC along a
        leading sample dimension. Specific times can be selected from the dataset
        by using ``start_indices``.

    Parameters:
        path: The path to the initial conditions dataset.
        engine: The engine used to open the dataset.
        start_indices: optional specification of the subset of
            initial conditions to use.
    """

    path: str
    engine: Literal["netcdf4", "h5netcdf", "zarr"] = "netcdf4"
    start_indices: StartIndices | None = None

    def get_dataset(self) -> xr.Dataset:
        ds = xr.open_dataset(
            self.path,
            engine=self.engine,
            decode_times=CFDatetimeCoder(use_cftime=True),
            decode_timedelta=False,
        )
        return self._subselect_initial_conditions(ds)

    def _subselect_initial_conditions(self, ds: xr.Dataset) -> xr.Dataset:
        if self.start_indices is None:
            ic_indices = slice(None, None)
        elif isinstance(self.start_indices, TimestampList):
            time_index = xr.CFTimeIndex(ds.time.values)
            ic_indices = self.start_indices.as_indices(time_index)
        else:
            ic_indices = self.start_indices.as_indices()
        # time is a required variable but not necessarily a dimension
        sample_dim_name = ds.time.dims[0]
        return ds.isel({sample_dim_name: ic_indices})
    
def get_initial_condition(
    ds: xr.Dataset, prognostic_names: Sequence[str]
) -> PrognosticState:
    """Given a dataset, extract a mapping of variables to tensors.
    and the time coordinate corresponding to the initial conditions.

    Args:
        ds: Dataset containing initial condition data. Must include prognostic_names
            as variables, and they must each have shape (n_samples, n_lat, n_lon).
            Dataset must also include a 'time' variable with length n_samples.
        prognostic_names: Names of prognostic variables to extract from the dataset.

    Returns:
        The initial condition and the time coordinate.
    """
    initial_condition = {}
    for name in prognostic_names:
        if len(ds[name].shape) != 3:
            raise ValueError(
                f"Initial condition variables {name} must have shape "
                f"(n_samples, n_lat, n_lon). Got shape {ds[name].shape}."
            )
        n_samples = ds[name].shape[0]
        initial_condition[name] = torch.tensor(ds[name].values).unsqueeze(dim=1)
    if "time" not in ds:
        raise ValueError("Initial condition dataset must have a 'time' variable.")
    initial_times = xr.DataArray(
        data=ds.time.values[:, None],
        dims=["sample", "time"],
    )
    if initial_times.shape[0] != n_samples:
        raise ValueError(
            "Length of 'time' variable must match first dimension of variables "
            f"in initial condition dataset. Got {initial_times.shape[0]} "
            f"and {n_samples}."
        )

    batch_data = BatchData.new_on_cpu(
        data=initial_condition,
        time=initial_times,
        horizontal_dims=["lat", "lon"],
    )
    return batch_data.get_start(prognostic_names, n_ic_timesteps=1)


@dataclasses.dataclass
class InferenceConfig:
    """
    Configuration for running inference.

    Parameters:
        experiment_dir: Directory to save results to.
        n_forward_steps: Number of steps to run the model forward for.
        checkpoint_path: Path to stepper checkpoint to load.
        logging: Configuration for logging.
        initial_condition: Configuration for initial condition data.
        forcing_loader: Configuration for forcing data.
        forward_steps_in_memory: Number of forward steps to complete in memory
            at a time.
        data_writer: Configuration for data writers.
        aggregator: Configuration for inference aggregator.
        stepper_override: Configuration for overriding select stepper configuration
            options at inference time (optional).
        allow_incompatible_dataset: If True, allow the dataset used for inference
            to be incompatible with the dataset used for stepper training. This should
            be used with caution, as it may allow the stepper to make scientifically
            invalid predictions, but it can allow running inference with incorrectly
            formatted or missing grid information.
    """

    experiment_dir: str
    logging_dir: str
    n_forward_steps: int
    checkpoint_path: str
    logging: LoggingConfig
    initial_condition: InitialConditionConfig
    forcing_loader: ForcingDataLoaderConfig
    forward_steps_in_memory: int = 10
    data_writer: DataWriterConfig = dataclasses.field(
        default_factory=lambda: DataWriterConfig()
    )
    aggregator: InferenceAggregatorConfig = dataclasses.field(
        default_factory=lambda: InferenceAggregatorConfig()
    )
    stepper_override: StepperOverrideConfig | None = None
    allow_incompatible_dataset: bool = False

    def __post_init__(self):
        if self.data_writer.time_coarsen is not None:
            validate_time_coarsen_config(
                self.data_writer.time_coarsen,
                self.forward_steps_in_memory,
                self.n_forward_steps,
            )

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(
        self, env_vars: dict | None = None, resumable: bool = False, **kwargs
    ):
        config = to_flat_dict(dataclasses.asdict(self))
        self.logging.configure_wandb(
            config=config, env_vars=env_vars, resumable=resumable, **kwargs
        )

    def load_stepper(self) -> Stepper:
        logging.info(f"Loading trained model checkpoint from {self.checkpoint_path}")
        return load_stepper(self.checkpoint_path, self.stepper_override)

    def load_stepper_config(self) -> StepperConfig:
        logging.info(f"Loading trained model checkpoint from {self.checkpoint_path}")
        return load_stepper_config(self.checkpoint_path, self.stepper_override)
    
    def get_data_writer(
            self,
            n_initial_conditions: int,
            timestep: datetime.timedelta,
            coords: Mapping[str, np.ndarray],
            variable_metadata: Mapping[str, VariableMetadata],
        ) -> DataWriter:
            return self.data_writer.build(
                experiment_dir=self.experiment_dir,
                n_initial_conditions=n_initial_conditions,
                n_timesteps=self.n_forward_steps,
                timestep=timestep,
                variable_metadata=variable_metadata,
                coords=coords,
                dataset_metadata=DatasetMetadata.from_env(),
            )
    def get_paired_data_writer(
            self,
            n_initial_conditions: int,
            timestep: datetime.timedelta,
            coords: Mapping[str, np.ndarray],
            variable_metadata: Mapping[str, VariableMetadata],
        ) -> PairedDataWriter:
            return self.data_writer.build(
                experiment_dir=self.experiment_dir,
                n_initial_conditions=n_initial_conditions,
                n_timesteps=self.n_forward_steps,
                timestep=timestep,
                variable_metadata=variable_metadata,
                coords=coords,
                dataset_metadata=DatasetMetadata.from_env(),
            )

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Folder to save to")
    parser.add_argument('-logging-dir', type=str, default=None,
                        help="Folder to save to")
    parser.add_argument('--model-dir', type=str,
                        help="Folder containing model data")
    parser.add_argument('--inference-config', type=str,
                        help="Path to the inference configuration file")
    parser.add_argument('--era5-dir', type=str,
                        help="Folder containing ERA5 data")
    parser.add_argument('--experiment-name', type=str, 
                        help="Identifying name of experiment")
    parser.add_argument('--num-steps-per-initialisation', type=int, required=True,
                        help='Number of autoregressive steps to run for every initialisation date')
    parser.add_argument('--start-datetime', type=str, required=True,
                        help='First target date, in YYYYMMDD-HH format')
    parser.add_argument('--end-datetime', type=str, default=None,
                        help='Final target date, in YYYYMMDD-HH format')
    parser.add_argument('--steps-between-initialisations', type=int, default=1,
                        help='Number of forecast steps between initialisations')
    parser.add_argument('--num-ensemble-members', type=int, default=1,
                        help='Number of ensemble members to use')
    parser.add_argument('--output-levels', type=int, nargs='+', default=[1000,850,500],
                        help="Which pressure levels to write to output. Needs to be a comma separated list, or -1 for all 37 levels.")  
    parser.add_argument('--output-vars', type=str, nargs='+', default=[
                                                                        '2m_temperature', 'total_precipitation_6hr', '10m_v_component_of_wind', 
                                                                        '10m_u_component_of_wind', 'specific_humidity', 'temperature', 'geopotential'
                                                                        ],
                        help="Which variables to write to output. Needs to be a space-separaeted list of values, or 'all' for all variables.")
    parser.add_argument('--save-every-n-steps', default=1, type=int,
                        help='Number of steps between saving outputs')
    parser.add_argument('--sst-input', default=None, choices=['forced', 'coupled'])
    parser.add_argument('--flux-model-path', type=str, default=None)
    parser.add_argument('--flux-data-config', type=str, default=None)
    parser.add_argument('--flux-model-config', type=str, default=None)
    parser.add_argument('--ocean-model-dir', type=str, default=None,
                        help='Directory in which ocean model is running')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    # Parse datetime arguments
    start_datetime = datetime.datetime.strptime(args.start_datetime, '%Y%m%d-%H')

    if args.end_datetime is not None:
        end_datetime = datetime.datetime.strptime(args.end_datetime, '%Y%m%d-%H')
    else:
        end_datetime = start_datetime
        
    
    config_overrides = [
        f"experiment_dir={os.path.join(args.output_dir, args.experiment_name)}",
        f"experiment_dir={args.logging_dir}",
        "n_forward_steps=" + str(args.num_steps_per_initialisation),
        "checkpoint_path=" + os.path.join(args.model_dir, "ace2_era5_ckpt.tar"),
        "stepper_override.ocean.interpolate=True",
        "initial_condition.path=" + os.path.join(args.model_dir, 'initial_conditions', f"ic_{start_datetime.year}.nc"),
        "forcing_loader.dataset.data_path=" + os.path.join(args.model_dir, 'forcing_data'),
        "forcing_loader.num_data_workers=" + str(2)
        ]
    ocean_config_overrides = []
    if args.sst_input == 'coupled':
        ocean_config_overrides += ["stepper_override.ocean.from_file.router_folder=" + args.ocean_model_dir,
                                   "stepper_override.ocean.from_file.polling_timeout=600",
                                   "stepper_override.ocean.from_file.sea_ice_fraction_name=sea_ice_fraction",
                                   "stepper_override.ocean.from_file.file_prefix=oce2atm",
                                   "stepper_override.ocean.from_file.file_suffix=_ace2_nemo",
                                   ]

    
    config_overrides += ocean_config_overrides

    config_data = prepare_config(args.inference_config, override=config_overrides)
    config = dacite.from_dict(
        data_class=InferenceConfig,
        data=config_data,
        config=dacite.Config(strict=True),
    )
    # Not sure how to set lists using the dotlist override, so do it manually here
    config.initial_condition.start_indices.times = [start_datetime.strftime("%Y-%m-%dT%H:%M:%S")]
    
    if args.sst_input == 'coupled':
        # IF using coupled setup, we can't look ahead more than one step
        config.forward_steps_in_memory = 1
        
        perturbation_config = {'type': 'from_file', 
                               'config': 
                                   {'data_directory': args.ocean_model_dir,
                                    'file_prefix': 'oce2atm',
                                    'file_suffix': '_ace2_nemo',
                                    'parameter_name_in_file': 'sea_ice_fraction',
                                    'parameter_name': 'sea_ice_fraction',
                                    'polling_timeout': 300
                                   }
        }
        
        config.forcing_loader.perturbations.perturbation_list = [dacite.from_dict(data_class=PerturbationSelector, data=perturbation_config, config=dacite.Config(strict=True))]      
    
    prepare_directory(config.experiment_dir, config_data)

    stepper_config = config.load_stepper_config()
    data_requirements = stepper_config.get_forcing_window_data_requirements(
        n_forward_steps=config.forward_steps_in_memory
    )
    logging.info("Loading initial condition data")
    inital_condition_ds = config.initial_condition.get_dataset()

    # # Write this initial condition to router directory as 0h file
    # output_dict = {k: inital_condition_ds[k].values for k in inital_condition_ds.data_vars}
    # with open(os.path.join(args.ocean_model_dir, "ace2_0h.pkl"), 'wb+') as ofh:
    #     pickle.dump(output_dict, ofh)
        
    initial_condition = get_initial_condition(
            inital_condition_ds, stepper_config.prognostic_names
        )

    # 
    stepper = config.load_stepper()
    stepper.set_eval()

    # This wraps around a Torch Data loader, so the best thing might be to construct
    # a different data loader. Or else mock the call to __getitem__ with data loader?
    logging.info("Initializing forcing data loader")
    data = get_forcing_data(
        config=config.forcing_loader,
        total_forward_steps=config.n_forward_steps,
        window_requirements=data_requirements,
        initial_condition=initial_condition,
        surface_temperature_name=stepper.surface_temperature_name,
        ocean_fraction_name=stepper.ocean_fraction_name
    )


    if not config.allow_incompatible_dataset:
        try:
            stepper.training_dataset_info.assert_compatible_with(data.dataset_info)
        except IncompatibleDatasetInfo as err:
            raise IncompatibleDatasetInfo(
                "Inference dataset is not compatible with dataset used for stepper "
                "training. Set allow_incompatible_dataset to True to ignore this "
                f"error. The incompatiblity found was: {str(err)}"
            ) from err

    variable_metadata = resolve_variable_metadata(
        dataset_metadata=data.variable_metadata,
        stepper_metadata=stepper.training_variable_metadata,
        stepper_all_names=stepper_config.all_names,
    )
    dataset_info = data.dataset_info.update_variable_metadata(variable_metadata)

    aggregator = config.aggregator.build(
        dataset_info=dataset_info,
        n_timesteps=config.n_forward_steps + stepper.n_ic_timesteps,
        output_dir=config.experiment_dir,
    )

    if args.output_dir is not None:
        writer = config.get_data_writer(
                n_initial_conditions=1,
                timestep=data.timestep,
                coords=data.coords,
                variable_metadata=variable_metadata,
            )
    else:
        writer = None

    run_inference(
            predict=stepper.predict_paired,
            data=data,
            writer=writer,
            aggregator=aggregator,
            record_logs=None,
        )