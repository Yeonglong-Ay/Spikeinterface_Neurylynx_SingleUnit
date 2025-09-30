#!/usr/bin/env python
# coding: utf-8

import subprocess, pathlib, os, shutil
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.curation as scur
import spikeinterface.widgets as sw
import numpy as np
from probeinterface import Probe

# Parallel config
global_job_kwargs = dict(n_jobs=4, chunk_duration="1s")
si.set_global_job_kwargs(**global_job_kwargs)

# Paths
REMOTE_PREFIX = "sunlab1:EMU_data/P06/Behav/SU/EXP24-25_Bari/Delayed_Discounting" # Change this to your file path
LOCAL_ROOT = pathlib.Path("./neurolnx")
LOCAL_ROOT.mkdir(exist_ok=True)

# List of filenames to process
ncs_filenames = [f"GA1-REC{i}.ncs" for i in range(1, 9)]
                # [f"GC1-LSTG{i}.ncs" for i in range(1, 9)] + \
                # [f"GB4-LMTG{i}.ncs" for i in range(1, 9)] + \
                # [f"GB3-LAH{i}.ncs" for i in range(1, 9)] + \
                # [f"GB2-LFSG{i}.ncs" for i in range(1, 9)] + \
                # [f"GB1-LPHG{i}.ncs" for i in range(1, 9)] + \
                # [f"GA4-LOPR-MI{i}.ncs" for i in range(1, 9)] + \
                # [f"GA3-LpSMA{i}.ncs" for i in range(1, 9)] + \
                # [f"GA2-ROF{i}.ncs" for i in range(1, 9)] + \
                # [f"GA1-REC{i}.ncs" for i in range(1, 9)]


for fname in ncs_filenames:
    print(f"\n--- Processing {fname} ---")
    local_ncs_path = LOCAL_ROOT / fname
    temp_folder = LOCAL_ROOT / f"temp_{fname.replace('.ncs', '')}"
    temp_folder.mkdir(exist_ok=True)

    # Download
    subprocess.run(["rclone", "copyto", f"{REMOTE_PREFIX}/{fname}", str(local_ncs_path), "--progress"], check=True)
    print(f"✅  Saved {fname} to {local_ncs_path}")

    # Move .ncs into its own folder
    temp_ncs_path = temp_folder / fname
    shutil.move(str(local_ncs_path), str(temp_ncs_path))

    # Load Neuralynx file
    recording = se.NeuralynxRecordingExtractor(folder_path=str(temp_folder))
    print(f"📏 Recording: {recording.get_num_channels()} channels @ {recording.get_sampling_frequency()} Hz")

    # Assign dummy probe
    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=np.array([[0.0, 0.0]]), shapes="circle", shape_params={"radius": 3})
    probe.set_device_channel_indices(np.array([0]))
    recording_geo = recording.set_probe(probe, in_place=False)

    # Bandpass filter
    recording_f = spre.bandpass_filter(recording_geo, freq_min=250, freq_max=3000, dtype="float32")
    recording_preprocessed = recording_f.save(format="binary")

    # Run spike sorter
    sorter_name = "kilosort4" 
    sorter_output = LOCAL_ROOT / f"sort_{fname.replace('.ncs', '')}"
    sorting_ks4 = ss.run_sorter(sorter_name=sorter_name,
                                recording=recording_preprocessed,
                                output_folder=sorter_output,
                                Th_universal=7.0,
                                keep_good_only=False)


                            
    ### SORTERS ###

    # sorter_name = "kilosort4" 
    # sorter_output = LOCAL_ROOT / f"sort_{fname.replace('.ncs', '')}"
    # sorting_ks4 = ss.run_sorter(sorter_name=sorter_name,
    #                             recording=recording_preprocessed,
    #                             output_folder=sorter_output,
    #                             singularity_image="./kilosort4-compiled-base-patched",
    #                             detect_threshold=4,
    #                             n_chan_bin=recording_preprocessed.get_num_channels(), 
    #                             verbose_console=True,
    #                             verbose_log=False)
     
    # sorter_name = "waveclus"
    # sorter_output = LOCAL_ROOT / f"sort_{fname.replace('.ncs', '')}"
    # sorting_wc = ss.run_sorter(sorter_name=sorter_name,
    #                             recording=recording_preprocessed,
    #                             output_folder=sorter_output,
    #                             singularity_image="./waveclus-compiled-base-patched",
    #                             detect_threshold=4,
    #                             min_clus=20,
    #                             stdmax=50,
    #                             keep_good_only=False)

    # sorter_name = "mountainsort5" 
    # sorter_output = LOCAL_ROOT / f"sort_{fname.replace('.ncs', '')}"
    # sorting_ms5 = ss.run_sorter(sorter_name="mountainsort5",
    #                             recording=recording_preprocessed,
    #                             output_folder=sorter_output,
    #                             singularity_image="./mountainsort5-compiled-base-patched",
    #                             detect_threshold=4)

    
    

    # Create analyzer
    analyzer_folder = LOCAL_ROOT / f"analyzer_{fname.replace('.ncs', '')}"
    sorting_analyzer = si.create_sorting_analyzer(
        sorting=sorting_ks4,
        recording=recording_geo,
        format="binary_folder",
        folder=str(analyzer_folder),
        n_jobs=1,
        progress_bar=True,
        chunk_duration="1s",
        overwrite=True
    )

    sorting_analyzer.compute("noise_levels")
    sorting_analyzer.compute("random_spikes")
    sorting_analyzer.compute("waveforms")
    sorting_analyzer.compute("templates") 
    sorting_analyzer.compute("spike_locations")
    sorting_analyzer.compute("spike_amplitudes") 
    sorting_analyzer.compute("correlograms")
    sorting_analyzer.compute("principal_components")
    sorting_analyzer.compute("quality_metrics")
    sorting_analyzer.compute("template_metrics")

    print(f"✅ Done processing {fname}")
