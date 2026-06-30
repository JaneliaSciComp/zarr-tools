import functools
import logging
import numpy as np
import re
import scipy.ndimage as ndi
import traceback
import zarr

from dask.array.core import normalize_chunks, slices_from_chunks
from dask.distributed import Client, as_completed
from .ngff.ngff_utils import (get_transformations_from_datasetpath,
                              get_spatial_axes, get_multiscales, add_new_dataset)
from toolz import partition_all
from typing import List, Tuple
from xarray_multiscale import windowed_mean, windowed_mode


logger = logging.getLogger(__name__)


def create_multiscale(dataset_store: zarr.storage.StoreLike,
                      dataset_path: str,
                      dataset_attrs: dict,
                      dataset_pattern: str,
                      data_type: str,
                      antialiasing:bool,
                      partition_size:int,
                      max_levels:int,
                      client: Client,
                      preserve_anisotropy:bool=False,
                      spatial_downsampling: List[Tuple[int,...]]|None=None):
    """
    Create a multiscale pyramid in the given Zarr group.
    """
    logger.info(f'Create multiscale for dataset {dataset_attrs}')
    dataset_regex = re.compile(dataset_pattern)
    pyramid_attrs = get_multiscales(dataset_attrs)
    logger.info(f'Current multiscale attributes {pyramid_attrs}')
    source_dataset_shape = dataset_attrs.get('array_shape', [])

    dataset_path_comps = dataset_path.strip('/').split('/')
    parent_dataset_path = '/'.join(dataset_path_comps[0:-1]).lstrip('/')
    if parent_dataset_path == dataset_path.strip('/'):
        # this happens only if the zarr container is actually a zarr array
        raise ValueError(f'Cannot generate multiscale pyramid for a top level zarr array - dataset_path is {dataset_path}')

    current_dataset_subpath = dataset_path_comps[-1]
    dataset_regex_match = dataset_regex.match(current_dataset_subpath)

    if dataset_regex_match is None:
        raise ValueError(f'Dataset path {dataset_path} does not match the scaled dataset regex: {dataset_pattern}')

    source_dataset_level = int(dataset_regex_match.group(1))
    source_dataset_scale, source_dataset_translation = get_transformations_from_datasetpath(
        pyramid_attrs, dataset_path,
        default_scale=(1,) * len(source_dataset_shape),
        default_translation=(0,) * len(source_dataset_shape),
    )
    dataset_blocksize = dataset_attrs.get('array_chunksize', [])
    logger.info((
        f'Start/Final dataset shapes extracted from {dataset_attrs}: '
        f'{source_dataset_shape}/{dataset_blocksize} '
    ))

    spatial_axes = get_spatial_axes(pyramid_attrs)

    def is_spatial_axis(axis:int) -> bool:
        return axis in spatial_axes.values()

    # we consider scalinng factors to be 1 for the first level from which we start downsizing
    scaling_factors = np.array([1] * len(source_dataset_shape))

    logger.info((
        f'Source level: {source_dataset_level}, '
        f'source dataset path: {dataset_path}, '
        f'shape: {source_dataset_shape} '
        f'source scale: {source_dataset_scale} '
        f'source translation: {source_dataset_translation} '
        f'preserve anisotropy: {preserve_anisotropy} '
    ))

    # open the multiscale group in append mode
    multiscale_group = zarr.open_group(store=dataset_store, path=parent_dataset_path, mode='a')
    dataset_arr = multiscale_group[current_dataset_subpath]
    current_level = source_dataset_level
    current_level_shape = dataset_arr.shape
    current_level_translation = source_dataset_translation
    current_level_scale = source_dataset_scale

    def next_level(match):
        level = int(match.group(1))
        return match.group(0).replace(str(level), str(level + 1), 1)

    spatial_axes_mask = [is_spatial_axis(i) for i in range(len(source_dataset_shape))]

    nlevels = 0
    while True:

        if max_levels > 0 and nlevels >= max_levels:
            break

        # all spatial dimensions are larger than the corresponding block size
        new_level_path = dataset_regex.sub(next_level, current_dataset_subpath)
        new_level = int(dataset_regex.match(new_level_path).group(1))

        if spatial_downsampling is not None:
            if nlevels >= len(spatial_downsampling):
                break
            level_spatial_downsampling = spatial_downsampling[nlevels]
            # extend to all dimensions if only spatial dims were provided
            ndims = len(current_level_shape)
            if len(level_spatial_downsampling) < ndims:
                level_downsampling = tuple(
                    level_spatial_downsampling[sum(spatial_axes_mask[:i])] if spatial_axes_mask[i] else 1
                    for i in range(ndims)
                )
            else:
                level_downsampling = level_spatial_downsampling
            # stop if all spatial dimensions are already smaller than the chunk size
            if all(not spatial_axes_mask[i] or current_level_shape[i] <= dataset_blocksize[i]
                   for i in range(len(current_level_shape))):
                break
        else:
            level_downsampling, found = _get_downsample_factors(
                new_level,
                current_level_shape,
                dataset_blocksize,
                tuple(current_level_scale),
                spatial_axes_mask,
                preserve_anisotropy=preserve_anisotropy,
            )
            if not found:
                break

        relative_scaling_factors  = np.array(level_downsampling)
        new_scaling_factors = scaling_factors * relative_scaling_factors
        new_level_scale = tuple(relative_scaling_factors * current_level_scale)
        new_level_translation = tuple([round((s * (dsf - 1) / 2) + tr, 3) if dsf > 1 else tr
                                                 for (s, dsf, tr)
                                                 in zip(current_level_scale, relative_scaling_factors, current_level_translation)])
        new_level_shape_array = (current_level_shape / relative_scaling_factors).astype(int)
        new_level_shape = tuple(map(int, new_level_shape_array))

        logger.info((
            f'New Level: {new_level}, '
            f'new level dataset path: {parent_dataset_path}/{new_level_path}, '
            f'new dataset shape (l{current_level} -> l{new_level}): {current_level_shape} -> {new_level_shape} '
            f'level downsampling factors (rel/abs): {relative_scaling_factors} / {new_scaling_factors} '
            f'level scale: {new_level_scale} '
            f'level translation: {new_level_translation} '
        ))

        pyramid_attrs = add_new_dataset(
            pyramid_attrs,
            new_level_path,
            scale_transform=new_level_scale,
            translation_transform=new_level_translation
        )

        logger.info((
            f'Create new dataset for level {new_level} at {parent_dataset_path}/{new_level_path} '
            f'pyramid_attrs -> {pyramid_attrs} '
        ))

        chunk_key_separator = ({'name': 'v2', 'separator': '/'}
                               if multiscale_group.metadata.zarr_format == 2
                               else None)
        sharding_kwargs = {}
        source_shards = getattr(dataset_arr, 'shards', None)
        if source_shards is not None:
            sharding_kwargs['shards'] = source_shards
        new_dataset_arr = multiscale_group.require_array(
            new_level_path,
            shape=new_level_shape,
            chunks=dataset_blocksize,
            dtype=dataset_arr.dtype,
            compressors=dataset_arr.compressors,
            fill_value=dataset_arr.fill_value,
            exact=True,
            chunk_key_encoding=chunk_key_separator,
            **sharding_kwargs,
        )

        downsample = functools.partial(
            _downsample,
            dataset_arr,
            new_dataset_arr,
            downsampling_factors=tuple(relative_scaling_factors),
            method='mode' if data_type == 'segmentation' else 'mean',
            antialiasing=antialiasing,
        )

        output_chunks = normalize_chunks(new_dataset_arr.metadata.chunk_grid.chunk_shape, shape=new_dataset_arr.shape)
        output_slices = slices_from_chunks(output_chunks)
        partitioned_output_slices = tuple(partition_all(partition_size, output_slices))
        logger.info(f'Partition level {new_level} with {len(output_slices)} blocks into {len(partitioned_output_slices)} partitions of up to {partition_size} blocks')

        for idx, part in enumerate(partitioned_output_slices):
            logger.info(f'Process level {new_level} partition {idx} ({len(part)} blocks)')

            res = client.map(downsample, part)
            for f, r in as_completed(res, with_results=True):
                if f.cancelled():
                    tb = f.traceback()
                    logger.error(f'Level {new_level} downsampling exception: {'\n'.join(traceback.format_tb(tb))}')
                    res = False
                else:
                    logger.debug(f'Level {new_level}: Finished writing blocks {r}')

            logger.info(f'Finished level {new_level} partition {idx}')

        dataset_arr = new_dataset_arr
        current_level = new_level
        current_level_shape = new_level_shape
        current_dataset_subpath = new_level_path
        scaling_factors = new_scaling_factors
        current_level_scale = new_level_scale
        current_level_translation = new_level_translation
        logger.info(f'Successfully downsampled level {current_level} to a {dataset_arr.shape} array ')
        nlevels = nlevels + 1

    # update multscales group attributes
    # I assume the dataset_attrs changed too when the multiscale extract from it were changed
    multiscale_group.attrs.update(dataset_attrs)


def _get_downsample_factors(target_level:int,
                            data_shape:Tuple[int, ...],
                            min_shape:Tuple[int, ...],
                            data_spacing:Tuple[float, ...],
                            spatial_axes_mask:List[bool],
                            anisotropy_threshold:float=0.1,
                            preserve_anisotropy:bool=False):
    if all(not spatial_axes_mask[i] or data_shape[i] <= min_shape[i] for i in range(len(data_shape))):
        return [1] * len(data_shape), False

    logger.info((
        f'Get downsample factors for level {target_level} from shape {data_shape} with spacing {data_spacing} '
    ))
    factors = [1] * len(data_shape)
    factors_changed = False
    min_spacing = min(s for i,s in enumerate(data_spacing) if spatial_axes_mask[i] )
    anisotropies = []
    for i, s in enumerate(data_spacing):
        # Calculate ratios of current spacing to all others
        # for example for 3D:  [(z/y, z/x),(y/z, y/x),(x/z, x/y)]
        if spatial_axes_mask[i]:
            anisotropy = data_spacing[i] / min_spacing
        else:
            anisotropy = 1
        anisotropies.append(anisotropy)

    for i in range(len(data_shape)):
        if spatial_axes_mask[i] and data_shape[i] > min_shape[i] // 2:
            if abs(1 - anisotropies[i]) < anisotropy_threshold or preserve_anisotropy:
                factors[i] = 2
                factors_changed = True

    logger.info((
        f'Computed downsample factors for level {target_level} from shape {data_shape} with spacing {data_spacing}: {factors} '
    ))
    return np.array(factors).astype(int), factors_changed


def _downsample(input:zarr.Array,
                output:zarr.Array,
                output_coords: Tuple[slice, ...],
                downsampling_factors:Tuple[int, ...]=(2,2,2),
                method:str='mean',
                antialiasing:bool=False):
    """
    Downsample source to target shape using the specified method.
    """

    input_coords = tuple(_multiply_slice(s, fact) for s, fact in zip(output_coords, downsampling_factors))
    logger.debug(f'Resample block {input_coords} -> {downsampling_factors}')
    input_block = input[input_coords]
    # only downsample source_data if it is not all 0s
    if not (input_block == 0).all():
        try:
            if method == 'mode':
                # this is the method used for segmentation data
                output[output_coords] = windowed_mode(input_block, window_size=downsampling_factors)
            else:
                # this is the method used for raw image data
                if antialiasing:
                    # blur data in chunk before downsampling to reduce aliasing of the image
                    # conservative Gaussian blur coeff: 2/2.5 = 0.8
                    sigma = [0 if factor == 1 else factor/2.5 for factor in downsampling_factors]
                    filtered_data = ndi.gaussian_filter(input_block, sigma=sigma)
                    output[output_coords] = windowed_mean(filtered_data, window_size=downsampling_factors)
                else:
                    output[output_coords] = windowed_mean(input_block, window_size=downsampling_factors)
        except Exception as e:
            logger.exception(f'Error resampling {input_block.shape} block at {input_coords} -> {downsampling_factors}', stack_info=True)
            raise e
        res = True
    else:
        res = False

    del input_block
    return output_coords, res


def _multiply_slice(s:slice, factor:int):
    return slice(s.start * factor, s.stop * factor, s.step)
