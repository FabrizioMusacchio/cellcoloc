# CellColoc example data

CellColoc example datasets are distributed through Zenodo. Download the archive from:

```text
https://doi.org/10.5281/zenodo.21603005
```

Place the downloaded archive contents in this `example_data/` folder.

## Dataset folders
The example-data collection is organized into two practical tiers:

| Folder | Purpose |
|---|---|
| `microglia_3D/` | Small multichannel CZI example dataset for quick CellColoc testing and tutorial use. |
| `dapi_stained_nuclei_2D/` | Redistributed 2D DAPI/nuclei example data from Rathar (2018) for lightweight testing and demonstrations. |
| `microglia_3D_full/` | Extended biological microglia CZI dataset used for the CellColoc preprint analysis. |
| `synthetic_benchmark_data_sharp/` | Main sharp filled-object synthetic benchmark dataset with ground truth and CellColoc results. |
| `synthetic_benchmark_data/` | Gaussian synthetic benchmark variant with ground truth and CellColoc results. |

Each dataset folder contains its own `README.md` with channel descriptions, intended use, and notes on included CellColoc result files.
