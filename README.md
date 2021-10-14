# LORACCS
This is the Python code to run the LOESS Radiometric Correction for Contiguous Scenes (LORACCS). 

LORACCS was developed to create seamless mosaics using Planet Dove imagery from the same day,
though it should work with other image sources, too.  It is mostly beneficial when trying to 
mosaic images from different Dove satellites. The scenes should overlap, and the overlapping 
area should be representative of the full scene  (for example, if the image is mostly forest,
the overlap area should have a lot of forest).

The paper corresponding to this work is published in the open source International 
Journal of Applied Earth Observations and Geoinformation:
https://authors.elsevier.com/sd/article/S0303-2434(20)30933-8

Of note: the current code removes the linear interpolation section (in the paper, section 3.2, 
list item 5) becuase updates to the LOESS package now allow for predictions. The code was 
updated to use this feature and improve the resulting model.

For questions or issues, please contact Sarah: wegmueller@wisc.edu

# Installation
LORACCS is formatted as a python class for ease of use, and can be run by simply downloading 
LORACCS.py and importing the class.  

The required packages are provided in the requirments.txt found in this repository.
In particular, you'll need rasterio, pandas and geopandas, matplotlib, numpy, shapely, and loess.

If using an Anaconda environment, the only package that requires pip install is the 
loess package (https://pypi.org/project/loess/)

```pip install loess```

The rest should be available via conda forge.  I've tested this with Python 3.7 and 3.8.

# Usage example with Dove imagery (using jupyter notebook or similar) 

Before using LORACCS, I HIGHLY recommend you mask the images first to get rid of any
bad pixels. For Dove imagery, this can be doing with the included UDM and UDM2 files.
This produces much better results.

```
from LORACCS import LORACCS

outdir = 'the filepath of the directory to which you would like the corrected image and associated outputs saved'
ref_img_fp = 'the filepath of the image to be used as reference'
tgt_img_fp = 'the filepath of the image to be corrected'
band_names = ['Blue', 'Red', 'Green', 'NIR']
max_spectra = [3000, 3000, 3000, 8000]

LORACCS(outdir, ref_img_fp, tgt_img_fp, band_names, max_spectra)
```
