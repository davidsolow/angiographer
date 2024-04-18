# Angiographer: Using Deep Learning to Identify Coronary Artery Stenosis

## Overview
This project will apply traditional CNN and U-Net models to attempt to automatically identify coronary artery stenosis using the [Aracde](https://www.nature.com/articles/s41597-023-02871-z) dataset.

## Dependencies
Dependencies are listed in `requirements.txt`. Note that this project was initially run on an Nvidia GPU. A standard Tensorflow installation may also be used. To do so, it is recommended to replace `tensorflow==2.15.0.post1` with `tensorflow==2.15.0`. For notebooks, a local jupyter environment is also required.

## How to use
First, data should should be downloaded [here](https://www.nature.com/articles/s41597-023-02871-z#Sec9). Extracted files within the `aracde` folder should be copied into the `raw_data` directory of this project.

Once data has been added, the `get_data.py` script should be run from a terminal within the `datasets` directory:

`python get_data.py`

This will load parquet and tensor files into the `datasets` directory. Once complete, EDA and models can be run using the notebooks in the `notebooks` directory.
