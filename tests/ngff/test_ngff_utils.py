import pytest

from zarr_tools.ngff.ngff_utils import create_ome_metadata


def test_create_ome_metadata_0_4_default_axes():
    metadata = create_ome_metadata(
        name='image',
        dataset_path='s0',
        axes=None,
        voxel_spacing=[3, 2, 1],
        voxel_translation=[6, 4, 2],
        image_ndims=3,
        ome_version='0.4',
    )

    assert 'ome' not in metadata
    multiscales = metadata['multiscales']
    assert len(multiscales) == 1

    multiscale = multiscales[0]
    assert multiscale['version'] == '0.4'
    assert multiscale['name'] == 'image'
    assert [a['name'] for a in multiscale['axes']] == ['z', 'y', 'x']
    assert all(a['type'] == 'space' and a['unit'] == 'micrometer' for a in multiscale['axes'])

    datasets = multiscale['datasets']
    assert len(datasets) == 1
    assert datasets[0]['path'] == 's0'

    transforms = {t['type']: t for t in datasets[0]['coordinateTransformations']}
    assert transforms['scale']['scale'] == [3, 2, 1]
    assert transforms['translation']['translation'] == [6, 4, 2]


def test_create_ome_metadata_0_4_with_channel_axis():
    metadata = create_ome_metadata(
        name='image',
        dataset_path='s0',
        axes=None,
        voxel_spacing=[2, 1, 1],
        voxel_translation=[0, 0, 0],
        image_ndims=4,
        default_unit='micrometer',
        ome_version='0.4',
    )

    multiscale = metadata['multiscales'][0]
    axis_names = [a['name'] for a in multiscale['axes']]
    assert axis_names == ['c', 'z', 'y', 'x']
    assert multiscale['axes'][0]['type'] == 'channel'

    transforms = {t['type']: t for t in multiscale['datasets'][0]['coordinateTransformations']}
    # scale/translation are padded with 1s/0s on the left to match image_ndims
    assert transforms['scale']['scale'] == [1, 2, 1, 1]
    assert transforms['translation']['translation'] == [0, 0, 0, 0]


def test_create_ome_metadata_0_5_default_axes():
    metadata = create_ome_metadata(
        name='image',
        dataset_path='s0',
        axes=None,
        voxel_spacing=[3, 2, 1],
        voxel_translation=[6, 4, 2],
        image_ndims=3,
        ome_version='0.5',
    )

    assert 'multiscales' not in metadata
    ome = metadata['ome']
    assert ome['version'] == '0.5'

    multiscales = ome['multiscales']
    assert len(multiscales) == 1

    multiscale = multiscales[0]
    # 0.5 multiscale objects do not carry their own version attribute
    assert 'version' not in multiscale
    assert multiscale['name'] == 'image'
    assert [a['name'] for a in multiscale['axes']] == ['z', 'y', 'x']

    datasets = multiscale['datasets']
    assert len(datasets) == 1
    assert datasets[0]['path'] == 's0'

    transforms = {t['type']: t for t in datasets[0]['coordinateTransformations']}
    assert transforms['scale']['scale'] == [3, 2, 1]
    assert transforms['translation']['translation'] == [6, 4, 2]


def test_create_ome_metadata_0_5_with_explicit_axes():
    axes = [
        {'name': 'c', 'type': 'channel'},
        {'name': 'z', 'type': 'space', 'unit': 'micrometer'},
        {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
        {'name': 'x', 'type': 'space', 'unit': 'micrometer'},
    ]

    metadata = create_ome_metadata(
        name='image',
        dataset_path='/some/nested/s0',
        axes=axes,
        voxel_spacing=[1, 1, 1, 1, 1],
        voxel_translation=[0, 0, 0, 0, 0],
        image_ndims=3,
        ome_version='0.5',
    )

    multiscale = metadata['ome']['multiscales'][0]
    # only the last 3 axes are kept when explicit axes are provided and image_ndims == 3
    assert [a['name'] for a in multiscale['axes']] == ['z', 'y', 'x']
    # dataset path is reduced to its last path component
    assert multiscale['datasets'][0]['path'] == 's0'
