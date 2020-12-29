# LORACCS
This is the code to run the LOESS Radiometric Correction for Contiguous Scenes (LORACCS). 

The paper corresponding to this work is published in the open source International 
Journal of Applied Earth Observations and Geoinformation.

LORACCS was developed to create seamless mosaics using Planet Dove imagery from the same day,
though it should work with other image sources, too.  It is mostly beneficial when trying to 
mosaic images from different Dove satellites. The scenes should overlap, and the overlapping 
area should be representative of the full scene  (for example, if the image is mostly forest,
the overlap area should have a lot of forest).

# Installation
LORACCS was formatted as a python class, and can be run by simply downloading 
LORACCS.py and importing the class.  

The required packages are provided in the requirments.txt found in this repository. 

If using an Anaconda environment, the only package that requires pip install is the 
loess package (https://pypi.org/project/loess/)

pip install loess

# Usage example (using jupyter notebook or similar) 

```from LORACCS import LORACCS```

```outdir = 'the filepath of the directory to which you would like the corrected image and associated outputs saved'
ref_img_fp = 'the filepath of the image to be used as reference'
tgt_img_fp = 'the filepath of the image to be corrected'

LORACCS(outdir, ref_img_fp, tgt_img_fp) ```





