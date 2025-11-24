# This module contains utility functions for concatenating zarr arrays
# from multiple sources into an output zarr array based on code written by Yurii Zubov.
#
# Original code by Yurii Zubov: https://github.com/yuriyzubov/recompress_zarr/blob/main/src/n5_to_zarr_consolidate.py
# Changes include the option of specifying the output "channel" for each input array

import logging
import zarr

from dask.array.core import slices_from_chunks, normalize_chunks
from dask.distributed import Client, as_completed
from toolz import partition_all
from typing import List, Tuple


logger = logging.getLogger(__name__)


def combine_arrays(input_zarrays: List[Tuple[zarr.Array, int, int]],
                   output_zarray:zarr.Array,
                   client: Client,
                   partition_size=100000):
    """
    Combine arrays
    """
    block_slices = slices_from_chunks(normalize_chunks(output_zarray.chunks, shape=output_zarray.shape))
    partitioned_block_slices = tuple(partition_all(partition_size, block_slices))
    logger.info(f'Partition {len(block_slices)} into {len(partitioned_block_slices)} partitions of up to {partition_size} blocks')

    nblocks = 0

    for idx, part in enumerate(partitioned_block_slices):
        logger.info(f'Process partition {idx} ({len(part)} blocks)')

        res = client.map(_copy_blocks, part, source_arrays=input_zarrays,output=output_zarray)
        npartition_blocks = 0
        for f, r in as_completed(res, with_results=True):
            if f.cancelled():
                exc = f.exception()
                logger.exception(f'Block processing exception: {exc}')
                res = False
            else:
                logger.debug(f'Finished writing blocks {r}')
                npartition_blocks = npartition_blocks + r

        logger.info(f'Finished partition {idx} - copied {npartition_blocks} blocks')
        nblocks = nblocks + npartition_blocks

    logger.info(f'Finished all {len(partitioned_block_slices)} - copied {nblocks} blocks')
    return nblocks


def _copy_blocks(coords, source_arrays=[], output=[]):
    # source_arrays have: (path, subpath, zarray, channel, timepoint)
    # the input arrays are all 3-D for now
    # this package cannot handle inputs other than 3-D
    nblocks = 0
    input_spatial_coords =coords[-3:] 
    for (_, _, arr, ch, tp) in source_arrays:
        if tp is not None:
            output_block_coords = (tp, ch) + input_spatial_coords
        else:
            output_block_coords = (ch,) + input_spatial_coords
        output[output_block_coords] = arr[input_spatial_coords]
        nblocks = nblocks + 1
    return nblocks
