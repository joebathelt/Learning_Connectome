# Association between structural connectome organisation and literacy and numeracy in children
This repository contains the scripts to generate FA connectome files and analyse these with graph theory. The main command line script executes the connectome construction from diffusion-weighted images. The analysis of the resulting connectivity matrices is carried out in the Jupyter Notebook file.

To generate FA connectome files:

```bash
python FA_Connectome.py --base_directory --out_directory --subject_list --ROI_file
```

input options:

--base_directory: directory containing raw data. The data needs to be organized in Brain Imaging Data Structure format (http://bids.neuroimaging.io/), i.e. {base_directory}/{subject}/dwi/{subject}_dwi.nii.gz', {base_directory}/{subject}/dwi/{subject}_dwi.bvec, {base_directory}/{subject}/dwi/{subject}_dwi.bval

--out_directory: path of the folder where the output will be directed

--subject_list: subject IDs separated by commata, e.g. CBU16001,CBU16002,CBU16003,etc.

--ROI_file: file containing a parcellation of the brain in MNI space, e.g. AAL atlas.

## Included scripts
- FA_Connectome: Python command line script that generates FA connectome matrices. In this matrix, the connection between two ROIs is expressed as the FA associated with streamlines that intersect with both ROIs.

- Structural_connectome_analysis.ipynb: Jupyter Notebook with analyses of structural connectome data using graph theoretical methods. I recommend using an online notebook viewer, e.g. http://nbviewer.jupyter.org/

## Dependencies
The scripts need a few neuroimaging packages to work:
- FMRIB Software Library (FSL): http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/
- MRTrix: http://jdtournier.github.io/mrtrix-0.2/

These python modules are also necessary (These can generally by installed using pip, e.g. pip install nipype):
- Diffusion Imaging in Python (DiPy): http://nipy.org/dipy/
- NeuroImaging Python Pipelines (NiPyPe): http://nipype.readthedocs.io/en/latest/index.html
- NiBabel: http://nipy.org/nibabel/
- NumPy: http://www.numpy.org/

## Overview of the workflow to create brain morphometry maps:
![alt tag](https://github.com/joebathelt/Learning_Connectome/blob/master/overview.png)


