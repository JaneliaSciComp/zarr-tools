import logging
import numpy as np

from ome_zarr_models.v04.image import (Dataset)
from pathlib import PurePosixPath
from typing import Dict, List, Tuple


logger = logging.getLogger(__name__)


def add_new_dataset(multiscales_attrs, dataset_path, scale_transform, translation_transform):
    datasets = multiscales_attrs.get('datasets', [])
    dataset_paths = [ds.get('path') for ds in datasets if ds.get('path', None)]

    logger.info(f'Add dataset path: {dataset_path} to {multiscales_attrs}')

    if scale_transform is not None:
        dataset = Dataset.build(path=dataset_path,
                                scale=scale_transform,
                                translation=translation_transform)

        existing_dataset = next((ds for ds in datasets if ds.get('path') == dataset_path), None)
        if existing_dataset is None:
            datasets.append(dataset.dict(exclude_none=True))

        existing_path = next((p for p in dataset_paths if p == dataset_path), None)
        if existing_path is None:
            dataset_paths.append(dataset_path)

        multiscales_attrs.update({
            'datasets': datasets,
            'paths': dataset_paths,
        })

    return multiscales_attrs


def create_ome_metadata(dataset_path, axes, voxel_spacing, voxel_translation, final_ndims,
                        default_unit='micrometer', ome_version='0.4'):
    # 0.5 is not compatible with 0.4 but 0.6+ should keep the backward compatibility
    if ome_version == '0.4':
        return _create_ome_metadata_0_4(dataset_path, axes, voxel_spacing, voxel_translation,
                                        final_ndims, default_unit=default_unit)
    else:
        return _create_ome_metadata_0_5(dataset_path, axes, voxel_spacing, voxel_translation,
                                        final_ndims, default_unit=default_unit, ome_version=ome_version)


def _create_ome_metadata_0_4(dataset_path, axes, voxel_spacing, voxel_translation, final_ndims,
                             default_unit='micrometer'):
    if not dataset_path:
        relative_dataset_path = ''
    else:
        # ignore leading '/'
        path_comps = [p for p in PurePosixPath(dataset_path).parts if p not in ('', '/')]
        relative_dataset_path = path_comps[-1]

    scale = ([1] if final_ndims == 4 else [1, 1]) + voxel_spacing
    translation = ([1] if final_ndims == 4 else [1, 1]) + voxel_translation
    if axes is None:
        multiscale_axes = [
            {
                "name": "z",
                "type": "space",
                "unit": default_unit,
            },
            {
                "name": "y",
                "type": "space",
                "unit": default_unit,
            },
            {
                "name": "x",
                "type": "space",
                "unit": default_unit,
            },
        ]
    else:
        multiscale_axes = axes[-3:]

    multiscale_axes.insert(0, {
        "name": "c",
        "type": "channel",
    })

    if final_ndims > 4:
        multiscale_axes.insert(0, {
            "name": "t",
            "type": "time",
        })

    dataset = {
        'path': relative_dataset_path,
        'coordinateTransformations': [
            {
                'type': 'scale',
                'scale': scale,
            },
            {
                'type': 'translation',
                'translation' : translation
            }
        ]
    }
    multiscales = {
        'multiscales': [
            {
                'axes': multiscale_axes,
                'datasets': [
                    dataset
                ],
                'version': '0.4',
                'name': '/',
            }
        ]
    }

    return multiscales


def _create_ome_metadata_0_5(dataset_path, axes, voxel_spacing, voxel_translation, final_ndims,
                             default_unit='micrometer', ome_version='0.5'):
    if not dataset_path:
        relative_dataset_path = ''
    else:
        # ignore leading '/'
        path_comps = [p for p in PurePosixPath(dataset_path).parts if p not in ('', '/')]
        relative_dataset_path = path_comps[-1]

    scale = ([1] if final_ndims == 4 else [1, 1]) + voxel_spacing
    translation = ([1] if final_ndims == 4 else [1, 1]) + voxel_translation
    if axes is None:
        multiscale_axes = [
            {
                "name": "z",
                "type": "space",
                "unit": default_unit,
            },
            {
                "name": "y",
                "type": "space",
                "unit": default_unit,
            },
            {
                "name": "x",
                "type": "space",
                "unit": default_unit,
            },
        ]
    else:
        multiscale_axes = axes[-3:]

    multiscale_axes.insert(0, {
        "name": "c",
        "type": "channel",
    })

    if final_ndims > 4:
        multiscale_axes.insert(0, {
            "name": "t",
            "type": "time",
        })

    dataset = {
        'path': relative_dataset_path,
        'coordinateTransformations': [
            {
                'type': 'scale',
                'scale': scale,
            },
            {
                'type': 'translation',
                'translation' : translation
            }
        ]
    }
    multiscales = {
        'multiscales': [
            {
                'axes': multiscale_axes,
                'datasets': [
                    dataset
                ],
                'name': '/',
            }
        ]
    }

    return {
        'ome': {
            'multiscales': multiscales
        }
    }


def get_axes_from_multiscales(multiscales_attrs):
    """
    Get multiscale axes if present or None otherwise
    """
    return multiscales_attrs.get('axes', None)


def get_axes(attrs):
    """
    Try first to retrieve the axes as if the attrs were multiscale attributes,
    otherwise retrieve the multiscales first and then retrieve the axes from the multiscales
    """
    return attrs.get('axes', get_axes_from_multiscales(get_multiscales(attrs)))


def get_axes_dictindex(attrs) -> Dict[str, int]:
    axes = get_axes(attrs)
    axes_dict = {}

    if axes is not None:
        for i, axis in enumerate(axes):
            axes_dict[axis['name'].lower()] = i

    return axes_dict


def get_spatial_axes(attrs) -> Dict[str, int]:
    """
    Get all spatial axes as a dictionary in which 
    the key is the axis name and the value is the axis index

    If the attributes are not valid OME metadata the returned value is an empty dictionary
    """
    axes = get_axes_dictindex(attrs)
    return {a:axes[a] for a in axes if a in ['x', 'y', 'z']}


def get_non_spatial_axes(attrs) -> Dict[str, int]:
    """
    Get all non-spatial axes (time and channel) as a dictionary in which
    the key is the axis name and the value is the axis index

    If the attributes are not valid OME metadata the returned value is an empty dictionary
    """
    axes = get_axes_dictindex(attrs)
    return {a:axes[a] for a in axes if a not in ['x', 'y', 'z']}


def get_dataset_at(multiscale_attrs, dataset_index):
    datasets = multiscale_attrs.get('datasets', [])
    if dataset_index < len(datasets):
        return datasets[dataset_index]
    else:
        return None


def get_dataset(multiscale_attrs, dataset_path):
    datasets = multiscale_attrs.get('datasets', [])
    dataset_path_comps = [c for c in dataset_path.split('/') if c]
    # lookup the dataset by path
    for ds in datasets:
        ds_path = ds.get('path', '')
        ds_path_comps = [c for c in ds_path.split('/') if c]
        logger.info((
            f'Compare current dataset path: {ds_path} ({ds_path_comps}) '
            f'with {dataset_path} ({dataset_path_comps}) '
        ))
        if (len(ds_path_comps) <= len(dataset_path_comps) and
            tuple(ds_path_comps) == tuple(dataset_path_comps[-len(ds_path_comps):])):
            # found a dataset that has a path matching a suffix of the dataset_subpath arg
            logger.info(f'Found dataset at {ds_path}')
            return ds

    return None


def get_dataset_transformations(dataset, default_scale=None, default_translation=None):
    """
    Get the scale and translation transformations from the multiscale attributes for the given dataset.
    """

    scale, translation = default_scale, default_translation
    coord_transformations = (dataset.get('coordinateTransformations', [])
                             if dataset is not None else [])
    for t in coord_transformations:
        if t['type'] == 'scale':
            scale = t['scale']
        elif t['type'] == 'translation':
            translation = t['translation']

    return scale, translation


def get_global_transformations(multiscale_attrs, default_scale=None, default_translation=None):
    coord_transformations = (multiscale_attrs.get('coordinateTransformations', []))
    scale, translation = default_scale, default_translation
    for t in coord_transformations:
        if t['type'] == 'scale':
            scale = t['scale']
        elif t['type'] == 'translation':
            translation = t['translation']

    return scale, translation


def get_transformations_from_datasetpath(multiscale_attrs, dataset_path, default_scale=None, default_translation=None):
    """
    Get the scale and translation transformations from the multiscale attributes for the dataset with the given path.
    """
    dataset_metadata = get_dataset(multiscale_attrs, dataset_path)

    scale, translation = default_scale, default_translation
    coord_transformations = (dataset_metadata.get('coordinateTransformations', [])
                             if dataset_metadata is not None else [])
    for t in coord_transformations:
        if t['type'] == 'scale':
            scale = t['scale']
        elif t['type'] == 'translation':
            translation = t['translation']

    return scale, translation


def get_datasets(multiscale_attrs):
    return multiscale_attrs.get('datasets', [])


def get_multiscales(attrs, index=0):
    """
    Get the multiscales attributes.
    """
    return attrs.get('multiscales', [{}])[index] # get the multiscale attributes at the specified index


def has_multiscales(attrs):
    """
    Get the multiscales attributes.
    """
    return 'multiscales' in attrs


def get_spatial_voxel_spacing(attrs) -> List[float] | None:
    multiscales = get_multiscales(attrs)
    dataset = get_dataset_at(multiscales, 0)
    voxel_resolution_values = None
    global_scale, _ = get_global_transformations(multiscales) if multiscales is not None else (None, None)
    dataset_scale, _ = get_dataset_transformations(dataset) if dataset is not None else (None, None)
    if global_scale is None and dataset_scale is None:
        if attrs.get('donwsamplingFactors'):
            # N5 at scale > S0
            pr = (np.array(attrs['pixelResolution']) * 
                np.array(attrs['downsamplingFactors']))
            voxel_resolution_values = pr[::-1].tolist()  # list of voxel spacings in zyx order
        elif attrs.get('pixelResolution'):
            # N5 at scale S0
            pr_attr = attrs.get('pixelResolution')
            if type(pr_attr) is list:
                pr = np.array(pr_attr)
                voxel_resolution_values = pr[::-1].tolist()  # list of voxel spacings in zyx order
            elif type(pr_attr) is dict:
                if pr_attr.get('dimensions'):
                    pr = np.array(pr_attr['dimensions'])
                    voxel_resolution_values = pr[::-1].tolist()  # list of voxel spacings in zyx order
    else:
        nspatial_axes = len(get_spatial_axes(multiscales)) or 3 # default to 3-D
        gscale_array = np.array(global_scale[-nspatial_axes:]) if global_scale else np.array((1,)*nspatial_axes)
        dscale_array = np.array(dataset_scale[-nspatial_axes:]) if dataset_scale else np.array((1,)*nspatial_axes)
        voxel_resolution_values = (gscale_array * dscale_array).tolist()

    return voxel_resolution_values
