# Documentation
Additional documentation can be found in the accompanied (work-in-progress) [documentation.pdf](documentation.pdf). This provides some general information, such as a description of the contrail detection pipeline.

# Getting Started

## Python Environment Setup

First we create our environment and install some general dependencies (e.g. pycontrails).

```
conda create --name contrails python=3.10
conda activate contrails
pip install -r requirements.txt
```

Then we need to setup the requirements necessary for the Global Meteor Network Code (see the original instructions [here](https://github.com/CroatianMeteorNetwork/RMS)). Note that the code in the included `/src/RMS` directory contains code from the original RMS repository, with some adaptations from Luc Busquin, and also changes from myself.

```
conda install -y -c conda-forge numpy scipy gitpython cython matplotlib paramiko
conda install -y -c conda-forge Pillow pyqtgraph'<=0.12.1'
conda install -y -c conda-forge ephem
conda install -y -c conda-forge imageio pandas
conda install -y -c conda-forge pygobject
conda install -y -c astropy astropy
conda install -y pyqt
pip install rawpy
pip install git+https://github.com/matejak/imreg_dft@master#egg=imreg_dft
```

The RMS code also requires some Cython code to be built. This can be done through the following:
```
cd src/RMS
python setup.py install
```

The contrail detection pipeline uses Segment Anything 2 (SAM2) which also needs to be installed. The original instructions for this can be found [here](https://github.com/facebookresearch/segment-anything-2).

This requires a machine that has a CUDA GPU, with Pytorch and Torchvision dependencies (follow the instructions [here](https://pytorch.org/get-started/locally/)). After Pytorch and Torchvision has been setup, SAM2 can be installed through the following:

```
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 & pip install -e .
```

Note: If you get the error `ImportError: cannot import name '_C' from 'sam2'`, then this provided solution worked for me.

```
In some systems, you may need to run `python setup.py build_ext --inplace` in the SAM 2 repo root as suggested in https://github.com/facebookresearch/segment-anything-2/issues/77.
```

</details>

SAM2 also requires a model checkpoint to be downloaded from [here](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt). This `sam2_hiera_large.pt` file should be downloaded to `/src/sam2_checkpoints`.

## Data Download
### ERA5
The ERA5 data can be downloaded using the script [here](src/data/era5/ERA5_downloader.ipynb).

### Flight Data
This was provided by Breakthrough Energy.

### GMN Data
The timelapse data is available publically [here](https://globalmeteornetwork.org/weblog/US/). However, the camera calibration is not available publically due to privacy reasons. Denis Vida (denis.vida@gmail.com) will need to give access for this. Once access has been given, the data can be downloaded through sftp, for example:

```
import subprocess

dates = ['20230820']
indiv_stations  = ['US001N']

for date in dates:
    for station in indiv_stations:
        result = subprocess.run(['sftp', f'jlin@gmn.uwo.ca:/home/{station.lower()}/files/processed/*_{date}_* ./data/gmn_tar_files/'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.returncode, result.stdout, result.stderr)
```

## Possible Problems

* The ERA5 data loading with pycontrails has had issues for me before. The solution for this can be found [here](https://github.com/contrailcirrus/pycontrails/issues/206).
* The segmentation script can have a memory leak issue, causing issues when running the script over large-scale datasets. This has not been resolved yet.