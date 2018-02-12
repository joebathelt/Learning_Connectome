#! /usr/bin/env python
import optparse
import os
import re
import sys


def main():
    p = optparse.OptionParser()

    p.add_option('--base_directory', '-b')
    p.add_option('--subject_list', '-s')
    p.add_option('--out_directory', '-o')

    options, arguments = p.parse_args()
    base_directory = options.base_directory
    out_directory = options.out_directory
    subject_list = options.subject_list
    subject_list = [subject for subject in subject_list.split(
        ',') if re.search('CBU', subject)]

    def get_ICV(subject_list, base_directory):
        #==============================================================
        # Loading required packages
        import nipype.interfaces.io as nio
        import nipype.pipeline.engine as pe
        import nipype.interfaces.utility as util
        from nipype.algorithms import misc
        from nipype import SelectFiles
        from nipype.interfaces import fsl
        from own_nipype import MAT2DET
        import os

        #====================================
        # Defining the nodes for the workflow

        # Getting the subject ID
        infosource = pe.Node(interface=util.IdentityInterface(
            fields=['subject_id']), name='infosource')
        infosource.iterables = ('subject_id', subject_list)

        # Getting the relevant diffusion-weighted data
        templates = dict(
            in_file='{subject_id}/anat/{subject_id}_T1w.nii.gz')

        selectfiles = pe.Node(SelectFiles(templates),
                              name="selectfiles")
        selectfiles.inputs.base_directory = os.path.abspath(base_directory)

        # Segment the image with FSL FAST
        fast = pe.Node(interface=fsl.FAST(), name='fast')
        fast.inputs.img_type = 1
        fast.inputs.no_bias = True

        # Select files from the FAST output
        GM_select = pe.Node(interface=util.Select(index=[1]), name='GM_select')
        WM_select = pe.Node(interface=util.Select(index=[2]), name='WM_select')

        # Calculate GM and WM volume with FSL stats
        GM_volume = pe.Node(interface=fsl.ImageStats(), name='GM_volume')
        GM_volume.inputs.op_string = '-M -V'

        WM_volume = pe.Node(interface=fsl.ImageStats(), name='WM_volume')
        WM_volume.inputs.op_string = '-M -V'

        flt = pe.Node(interface=fsl.FLIRT(), name='flt')
        flt.inputs.reference = os.environ[
            'FSLDIR'] + '/data/standard/MNI152_T1_1mm_brain.nii.gz'

        mat2det = pe.Node(interface=MAT2DET(), name='mat2det')

        # Create an output csv file
        addrow = pe.Node(interface=misc.AddCSVRow(), name='addrow')
        addrow.inputs.in_file = out_directory + 'volume_results.csv'

        #====================================
        # Setting up the workflow
        get_ICV = pe.Workflow(name='get_ICV')
        get_ICV.connect(infosource, 'subject_id', selectfiles, 'subject_id')
        get_ICV.connect(selectfiles, 'in_file', flt, 'in_file')
        get_ICV.connect(flt, 'out_matrix_file', mat2det, 'in_matrix')
        get_ICV.connect(infosource, 'subject_id', mat2det, 'subject_id')
        get_ICV.connect(infosource, 'subject_id', fast, 'out_basename')
        get_ICV.connect(selectfiles, 'in_file', fast, 'in_files')
        get_ICV.connect(fast, 'partial_volume_files', GM_select, 'inlist')
        get_ICV.connect(GM_select, 'out', GM_volume, 'in_file')
        get_ICV.connect(fast, 'partial_volume_files', WM_select, 'inlist')
        get_ICV.connect(WM_select, 'out', WM_volume, 'in_file')
        get_ICV.connect(infosource, 'subject_id', addrow, 'MRI.ID')
        get_ICV.connect(GM_volume, 'out_stat', addrow, 'GM_volume')
        get_ICV.connect(WM_volume, 'out_stat', addrow, 'WM_volume')

        #====================================
        # Running the workflow
        get_ICV.base_dir = os.path.abspath(out_directory)
        get_ICV.write_graph()
        get_ICV.run()

    get_ICV(subject_list, base_directory)

if __name__ == '__main__':
    # main should return 0 for success, something else (usually 1) for error.
    sys.exit(main())
