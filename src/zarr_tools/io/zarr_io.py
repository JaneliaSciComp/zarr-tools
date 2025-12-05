import logging
import os
import re
import zarr

from ..ngff.ngff_utils import (get_axes, get_dataset, get_datasets,
                               get_multiscales, has_multiscales,
                               get_axes_from_multiscales, get_global_transformations,
                               get_dataset_transformations)
from typing import List,Tuple
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


def _get_compressor(compressor: str | None, compression_opts: dict, zarr_format: int):
    logger.info(f'Get compressor: {compressor}: {compression_opts} for zarr format {zarr_format}')
    if compressor is None:
        return None

    if zarr_format == 3:
        import zarr.codecs as zv3codecs

        compressor_map = {
            'packbits': zv3codecs.PackBits,
            'zstd': zv3codecs.Zstd,
            'zlib': zv3codecs.Zlib,
            'gzip': zv3codecs.GZip,
            'bz2': zv3codecs.BZ2,
            'lzma': zv3codecs.LZMA,
            'lz4': zv3codecs.LZ4,
            'blosc': zv3codecs.Blosc,
        }
        codec_cls = compressor_map.get(compressor.lower())
        if codec_cls is None:
            raise ValueError(f'Unsupported compressor {compressor} for zarr format {zarr_format}')
        return codec_cls(**compression_opts)
    else:
        import numcodecs as zv2codecs

        return zv2codecs.get_codec({'id': compressor, **compression_opts})


def _compressor_kwargs(codec, zarr_format: int):
    if zarr_format == 3:
        # zarr v3 expects a tuple of BytesBytesCodec (or an empty tuple)
        compressors = None if codec is None else (codec,)
        return {'compressors': compressors}
    else:
        return {'compressor': codec}


def create_zarr_array(container_path:str,
                      array_subpath:str,
                      shape:Tuple[int],
                      chunks:Tuple[int],
                      dtype:str,
                      compressor:str|None=None,
                      compression_opts:dict={},
                      overwrite=False,
                      parent_array_attrs={},
                      zarr_format=2, # default zarr v2 format
                      **array_attrs):

    real_container_path = os.path.realpath(container_path)
    store = zarr.storage.LocalStore(real_container_path)

    codec = _get_compressor(compressor, compression_opts, zarr_format)
    compressor_args = _compressor_kwargs(codec, zarr_format)

    chunk_key_separator = {'name': 'v2', 'separator': '/'} if zarr_format == 2 else None

    if array_subpath:
        logger.info((
            f'Create array {container_path}:{array_subpath} '
            f'compressor={compressor}, shape: {shape}, chunks: {chunks} '
            f'parent attrs: {parent_array_attrs} '
            f'array attrs: {array_attrs} '
        ))

        root_group = zarr.open_group(store=store, mode='a', zarr_format=zarr_format)
        if overwrite:
            # create an array dataset no matter whether it exists or not
            current_shape = shape
            zarray = root_group.create_array(
                array_subpath,
                shape=current_shape,
                chunks=chunks,
                dtype=dtype,
                overwrite=True,
                chunk_key_encoding=chunk_key_separator,
                **compressor_args,
            )
        else:
            if array_subpath in root_group:
                # if the array already exists, get its shape
                zarray = root_group[array_subpath]
                current_shape = zarray.shape
                logger.info((
                    f'Array {container_path}:{array_subpath} '
                    f'already exists with shape {current_shape} '
                ))
            else:
                # this is a new array
                current_shape = shape
                zarray = root_group.create_array(
                    array_subpath,
                    shape=current_shape, # use the current shape
                    chunks=chunks,
                    dtype=dtype,
                    overwrite=True,
                    chunk_key_encoding=chunk_key_separator,
                    **compressor_args,
                )
            _resize_zarr_array(zarray, shape)
            _update_parent_attrs(root_group, array_subpath, parent_array_attrs)
            zarray.attrs.update(array_attrs)
            return zarray
    else:
        # the zarr container is the array
        if overwrite:
            current_shape = shape
            zarray = zarr.create_array(
                store=store,
                shape=current_shape,
                chunks=chunks,
                dtype=dtype,
                overwrite=True,
                zarr_format=zarr_format,
                chunk_key_encoding=chunk_key_separator,
                **compressor_args,
            )
        elif zarr.storage.contains_array(store):
            # the array already exists
            zarray = zarr.open(store=store, mode='a')
            current_shape = zarray.shape
            _resize_zarr_array(zarray, shape)
        else:
            current_shape = shape
            zarray = zarr.create_array(
                store=store,
                shape=current_shape,
                chunks=chunks,
                dtype=dtype,
                zarr_format=zarr_format,
                chunk_key_encoding=chunk_key_separator,
                **compressor_args,
            )
        zarray.attrs.update(array_attrs)
        return zarray



def create_zarr_group(container_path:str, group_subpath:str, 
                      store_name:str|None=None,
                      zarr_format:int=2):
    real_container_path = os.path.realpath(container_path)
    if store_name == 'n5':
        store = zarr.N5Store(real_container_path)
    else:
        store = zarr.storage.LocalStore(real_container_path)
    root_group = zarr.open_group(store=store, mode='a', zarr_format=zarr_format)
    if group_subpath in root_group:
        # get group
        g = root_group[group_subpath]
    else:
        # create the group
        g = root_group.create_group(group_subpath)
    return g


def open_zarr_store(data_path:str, data_subpath:str, data_store_name:str|None=None, mode:str='r') -> Tuple[zarr.storage.StoreLike, dict]:
    try:
        zarr_container, zarr_subpath = _get_data_store(data_path, data_subpath, data_store_name)

        logger.info(f'Open zarr container: {zarr_container} ({zarr_subpath})')
        data_container = zarr.open(store=zarr_container, mode=mode)
        multiscales_zgroup, dataset_subpath, multiscales_attrs  = _lookup_ome_multiscales(data_container, zarr_subpath)

        if multiscales_zgroup is not None:
            logger.info((
                f'Open OME ZARR {zarr_container.root}:{zarr_subpath} '
            ))
            return _open_ome_zarr_store(multiscales_zgroup, dataset_subpath, multiscales_attrs)
        else:
            logger.info(f'Open Simple ZARR {data_container.path}:{zarr_subpath}')
            return _open_simple_zarr_store(data_container, zarr_subpath)
    except Exception as e:
        logger.error(f'Error opening {data_path}:{data_subpath} {e}')
        raise e


def _get_data_store(data_path, data_subpath, data_store_name):
    """
    This methods adjusts the container and dataset paths such that
    the container paths always contains a .attrs file
    """
    path_comps = os.path.splitext(data_path)
    ext = path_comps[1]
    if (ext is not None and ext.lower() == '.n5' or data_store_name == 'n5'):
        # N5 container path is the same as the data_path
        # and the subpath is the dataset path
        logger.info(f'Create N5 store for {data_path}: {data_subpath}')
        return zarr.N5Store(data_path), data_subpath

    logger.info(f'Create ZARR store for {data_path}: { data_subpath}')
    dataset_path_arg = data_subpath if data_subpath is not None else ''
    dataset_comps = [c for c in dataset_path_arg.split('/') if c]
    dataset_comps_index = 0

    # Look for a valid container path - must contain
    while dataset_comps_index < len(dataset_comps):
        container_subpath = '/'.join(dataset_comps[0:dataset_comps_index])
        container_path = f'{data_path}/{container_subpath}'
        if (os.path.exists(f'{container_path}/.zgroup') or
            os.path.exists(f'{container_path}/.zattrs') or
            os.path.exists(f'{container_path}/.zarray') or
            os.path.exists(f'{container_path}/attributes.json') or
            os.path.exists(f'{container_path}/zarr.json')):
            break
        dataset_comps_index = dataset_comps_index + 1

    appended_container_path = '/'.join(dataset_comps[0:dataset_comps_index])
    container_path = f'{data_path}/{appended_container_path}'
    new_subpath = '/'.join(dataset_comps[dataset_comps_index:])

    logger.debug(f'Found zarr container at {container_path}:{new_subpath}')
    return zarr.storage.LocalStore(container_path), new_subpath


def _lookup_ome_multiscales(data_container, data_subpath):
    logger.debug(f'lookup OME multiscales group within {data_subpath}')
    dataset_subpath_arg = data_subpath if data_subpath is not None else ''
    dataset_comps = [c for c in dataset_subpath_arg.split('/') if c]

    dataset_comps_index = 1
    while dataset_comps_index < len(dataset_comps):
        container_item_subpath = '/'.join(dataset_comps[0:dataset_comps_index])
        container_item = data_container[container_item_subpath]
        container_item_attrs = container_item.attrs.asdict()

        if has_multiscales(container_item_attrs):
            logger.debug(f'Found multiscales at {container_item_subpath}: {container_item_attrs}')
            # found a group that has attributes which contain multiscales list
            return container_item, '/'.join(dataset_comps[dataset_comps_index:]), container_item_attrs
        else:
            dataset_comps_index = dataset_comps_index + 1

    # if no multiscales have found - look directly under root
    data_container_attrs = data_container.attrs.asdict()
    if has_multiscales(data_container_attrs):
        logger.debug(f'Found multiscales directly under root: {data_container_attrs}')
        # the container itself has multiscales attributes
        return data_container, data_subpath, data_container_attrs
    else:
        return None, None, {}


def _open_ome_zarr_store(multiscales_zgroup, dataset_subpath, attrs) -> Tuple[zarr.storage.StoreLike, dict]:

    multiscale_metadata = get_multiscales(attrs)

    dataset_metadata = get_dataset(multiscale_metadata, dataset_subpath)

    if dataset_metadata is None:
        logger.info(f'No dataset was found using {dataset_subpath}')
        if dataset_subpath in multiscales_zgroup:
            logger.debug(f'Dataset {dataset_subpath} found in the {multiscales_zgroup.path} container but not in metadata')
            # lookup a dataset in the metadata that could potentially have the same scale
            # and use the transformations from that one
            dataset_comps = [c for c in dataset_subpath.split('/') if c]
            dataset_index_comp = dataset_comps[-1]
            datasets = get_datasets(multiscale_metadata)
            dataset_index = _extract_numeric_comp(dataset_index_comp)
            if dataset_index < len(datasets):
                dataset_with_matching_scale = datasets[dataset_index]
                dataset_metadata = {
                    'path': dataset_subpath,
                    'coordinateTransformations': dataset_with_matching_scale.get('coordinateTransformations', []),
                }
            else:
                dataset_metadata = {
                    'path': dataset_subpath,
                }
        else:
            # could not find the dataset in the group
            raise ValueError(f'No dataset found for {dataset_subpath}')

    dataset_path = dataset_metadata.get('path')
    logger.info(f'Get dataset using path: {dataset_path}')
    za = multiscales_zgroup[dataset_path] if dataset_path else multiscales_zgroup
    global_scale, global_translation = get_global_transformations(multiscale_metadata)
    dataset_scale, dataset_translation = get_dataset_transformations(dataset_metadata)
    _set_array_attrs(attrs, za.store, za.store_path, za.shape, za.dtype, za.chunks,
                     axes=get_axes_from_multiscales(multiscale_metadata),
                     global_scale=global_scale, global_translation=global_translation,
                     dataset_scale=dataset_scale, dataset_translation=dataset_translation)

    return za.store, attrs


def _extract_numeric_comp(v):
    match = re.match(r'^(\D*)(\d+)$', v)
    if match:
        return int(match.groups()[1])
    else:
        raise ValueError(f'Invalid component: {v}')


def _open_simple_zarr_store(data_container, data_subpath) -> Tuple[zarr.storage.StoreLike, dict]:
    if not data_subpath or data_subpath == '.':
        # the input parameter is an array
        shape = data_container.shape
        dtype = data_container.dtype
        chunks = data_container.chunks

        attrs = data_container.attrs.asdict()

        _set_array_attrs(attrs, data_container.store, data_container.store_path, shape, dtype, chunks)
        return data_container.store, attrs
    else:
        a = data_container[data_subpath]
        shape = a.shape
        dtype = a.dtype
        chunks = a.chunks

        dataset_comps = [c for c in data_subpath.split('/') if c]
        parent_group_subpath = '/'.join(dataset_comps[:-1])
        if parent_group_subpath == '':
            parent_group = data_container
        else:
            parent_group = data_container[parent_group_subpath]

        attrs = parent_group.attrs.asdict()

        _set_array_attrs(attrs, a.store, a.store_path, shape, dtype, chunks)
        return a.store, attrs


def _set_array_attrs(attrs, store, array_store_path, shape, dtype, chunks,
                     axes=None, dataset_scale=None, dataset_translation=None,
                     global_scale=None, global_translation=None,):
    """
    Add useful datasets attributes from the array attributes:
    shape, ndims, data_type, chunksize
    """
    array_store_path_str = str(array_store_path) # get store path as a string
    array_container_path = urlparse(str(store)).path
    array_subpath = urlparse(array_store_path_str).path.removeprefix(array_container_path).strip('/')

    attrs.update({
        'array_axes': axes,
        'array_storepath': str(store),
        'array_subpath': array_subpath,
        'array_shape': shape,
        'array_ndim': len(shape),
        'array_dtype': dtype.name,
        'array_chunksize': chunks,
        'array_transforms': {
            'scale': dataset_scale,
            'translation': dataset_translation,
        },
        'global_transforms': {
            'scale': global_scale,
            'translation': global_translation,
        }
    })
    return attrs


def _resize_zarr_array(zarray, new_shape):
    """
    Resize the array to fit the new shape
    """
    if zarray.shape != new_shape:
        logger.info(f'Resize array from {zarray.shape} to {new_shape}')
        zarray.resize(new_shape)


def _update_parent_attrs(root_group, array_subpath, parent_attrs):
    if array_subpath:
        # create the parent group if needed
        parent_array_path = os.path.dirname(array_subpath)
        parent_group = (root_group if not parent_array_path
                        else root_group.require_group(parent_array_path))
    else:
        parent_group = root_group

    parent_group.attrs.update(parent_attrs)


def read_zarr_block(arr, metadata,
                    timeindex: int|None,
                    channelindex:int|List[int]|None,
                    block_coords: Tuple|None):
    """
    Read a data block from the specified coordinates.
    """
    ndim = arr.ndim
    # if there are fewer coordinates than the array dimension,
    # extend the block coordinates up to the number of array dimensions
    if block_coords is None:
        input_block_coords = (slice(None,None),) * ndim
    elif len(block_coords) < ndim:
        input_block_coords = (slice(None,None),) * (ndim - len(block_coords)) + block_coords
    else:
        input_block_coords = block_coords

    selector = []
    selection_exists = False

    axes = get_axes(metadata) or []

    for ai, a in enumerate(axes):
        if a.get('type') == 'time':
            if timeindex is not None:
                selector.append(timeindex)
                selection_exists = True
            else:
                selector.append(input_block_coords[ai])
        elif a.get('type') == 'channel':
            if channelindex is None or ch == []:
                selector.append(input_block_coords[ai])
            else:
                selector.append(channelindex)
                selection_exists = True
        else:
            selector.append(input_block_coords[ai])

        selection_exists = (selection_exists or
                            input_block_coords[ai].start is not None or
                            input_block_coords[ai].stop is not None)

    if selection_exists:
        try:
            # try to select the data using the selector
            block_slice_coords = tuple(selector)
            logger.debug(f'Get block at {block_slice_coords}')
            return arr[block_slice_coords]
        except Exception  as e:
            logger.exception(f'Error selecting data with selector {tuple(selector)}')
            raise e
    else:
        return arr
