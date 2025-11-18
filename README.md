# zarr-tools
Command line zarr tools used by various nextflow modules to run multiscale or to combine channels from multiple ZARR arrays in a single OME-ZARR

# Concatenating multiple Zarr arrays example
```
python -m zarr_tools.cli.main_combine_arrays \
    --output <outputlocation>/output.zarr --output-dataset 0 \
    --output-chunks 256,256,128 \
    --array-params \
    <input1_location>/input1.zarr:<input1-dataset>:<output_channel_index_for_input1> \
    <input2_location>/input2.zarr:<input2-dataset>:<output_channel_index_for_input2> \
    <input3_location>/input3.zarr:<input2-dataset>:<output_channel_index_for_input3> \
    --compressor blosc \
    --worker-cpus 1 \
    --dask-config configs/dask-config.yml \
    --partition-size 5000 \
    --local-dask-workers 80
```

# Generate multiscale example
```
python -m zarr_tools.cli.main_multiscale \
    -i <data_location>/data.zarr --input-subpath <scale_0_dataset_subpath> \
    --local-dask-workers 4 \
    --data-type [raw|segmentation] \
    --dataset-pattern ".*(\d)+"
```
The downsample interpolation is done differently for labels zarr than raw data, that's the reason for `--data-type` flag.
`--dataset-pattern` typically is not needed - the default should be sufficient - but if provided it is used to ensure the different scale level datasets follow the same pattern.