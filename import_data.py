import numpy as np
import pandas as pd
import csv
from scipy import constants as con
from pathlib import Path, PurePath

trpl_laser_reference_file = PurePath(Path().cwd(), 'ref_files/2024_03_12_TRPL_Laserpower.txt')
plqe_laser_reference_file = PurePath(Path().cwd(), 'ref_files/PLQE_laser.txt')


def unpack_trpl_info(path, trpl_laser_reference_file, laser_intensity, sample_absorbtance, sample_thickness):

    encoding = 'mac_roman'

    # load laser calibration file
    wl400, wl505, wl630 = np.loadtxt(trpl_laser_reference_file, unpack=True, skiprows=1, encoding=encoding)

    # import trpl measurement file
    rows = []
    with open(path,'r',encoding=encoding ) as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            rows.extend(row)

    # generate warning if pile-up rate is more than 5%
    signal_rate = float([i for i in rows if 'Signal_Rate' in i][0].split(' : ')[1][:-3])  # in cps
    sync_frequency = float([i for i in rows if 'Sync_Frequency' in i][0].split(' : ')[1][:-2])  # in Hz
    pile_up = signal_rate / sync_frequency
    if pile_up > 0.05:
        print(path, 'has pile-up rate above 5%. Use of measurement not recommended (see picoquant docs for further info).')
    
    # extract excitation attenuation and laser wavelength
    att = [i for i in rows if 'Exc_Attenuation' in i][0].split(' : ')[1][:-4]
    if att == 'open':
        attenuation = 1
    else:
        attenuation = float(att[0:-1]) / 100

    wavelength = float([i for i in rows if 'Exc_Wavelength' in i][0].split(' : ')[1][:-2])  # in nm

    # Using attenuation and laser wavelength, calculate laser fluence (photons cm-2 pulse-1)
    if wavelength == 397.7:
        laser_fluence = (wl400[0] * laser_intensity + wl400[1]) * attenuation # (photons cm-2 pulse-1)

    elif wavelength == 505.5:
        laser_fluence = (wl505[0] * laser_intensity + wl505[1]) * attenuation # (photons cm-2 pulse-1)

    elif wavelength == 633.8:
        laser_fluence = (wl630[0] * laser_intensity + wl630[1]) * attenuation # (photons cm-2 pulse-1)

    # Output excitation density (photons cm-3 pulse-1)
    exc_density = laser_fluence * sample_absorbtance / (sample_thickness * 1e-7)  # photons cm-3 pulse-1

    return exc_density


def import_trpl_data(path, mode, noise_threshhold):

    encoding = 'mac_roman'

    ## Data is imported

    time_raw, counts_raw = [[],[]]
    with open(path,'r',encoding=encoding) as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if not row:
                pass
            elif row[0] == 'Time [ns]':
                csvstart = reader.line_num
    
    with open(path,'r',encoding=encoding) as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if reader.line_num > csvstart:
                time_raw.append(float(row[0]))
                counts_raw.append(float(row[1]))

    time_raw=np.array(time_raw)[:-5] #cut off last 5 data points bc they can be weird
    counts_raw=np.array(counts_raw)[:-5]

    # Define t=0 at the highest counts (expected to be the start of the pulse)
    if mode == 'burst':
        # find rise by determining largest change over 5 succesive data points
        t0_index = np.nanargmax( [np.var(counts_raw[50+i:50+i+5]) for i in range(len(counts_raw)-5-50)] ) + 50
    else:
        t0_index = np.nanargmax(counts_raw)
    t0 = time_raw[t0_index]
    time_processed = (time_raw - t0)[t0_index:]

    # Calculate signal to noise (which is the average over the signal before the pulse)
    signal_to_noise = counts_raw[t0_index:] / np.nanstd(counts_raw[t0_index:])
    noise_cutoff_index = int(np.argwhere(signal_to_noise < noise_threshhold)[0][0] + t0_index)
    noise_cutoff = time_processed[noise_cutoff_index]

    trpl_range_start = 0 
    noise_mask = (time_processed >= trpl_range_start) & (time_processed <= noise_cutoff)

    # normalize trpl data
    counts = np.where(counts_raw[t0_index:] >= 0, counts_raw[t0_index:], 0) # set negative values to zero
    counts_normalized = counts/np.nanmax(counts)

    return time_processed, counts_normalized, noise_mask


def import_plqe_data(path, sample_name, plqe_laser_wl, plqe_laser_reference_file, sample_absorbtance, sample_thickness):

    PLQE_data = pd.read_csv(path)
    print(sample_name)

    for col in PLQE_data.columns:
        if (sample_name  in col) and ('PLQE' in col):
            column_index = PLQE_data.columns.get_loc(col)
            PLQE = np.array(PLQE_data[col])
            power = np.array(PLQE_data.iloc[:,(column_index-1)])

    # load laser calibration file
    plqe_calibration = pd.read_csv(plqe_laser_reference_file, sep='\t')
    plqe_power_conversion_factor = plqe_calibration.loc[plqe_calibration['laser']==int(plqe_laser_wl), 'power conversion factor'].values[0]
    plqe_laser_ss = plqe_calibration.loc[plqe_calibration['laser']==int(plqe_laser_wl), 'spotsize (cm2)'].values[0]

    # convert laser fluence to generation rate
    fluences = ((power/plqe_power_conversion_factor)/plqe_laser_ss) / (con.h*con.c/(plqe_laser_wl*1e-9)) # photons s-1 cm-2
    generation_rates = (fluences* sample_absorbtance/(sample_thickness * 1e-7)) # photons s-1 cm-3

    return generation_rates[~np.isnan(PLQE)], PLQE[~np.isnan(PLQE)]


def import_all_data(directory, input_df, trpl_laser_reference_file, noise_threshhold, plqe_laser_reference_file):

    data = {}
    measurements = input_df['measurement']

    for i in range(len(measurements)):

        sample_index = list(input_df.index)[i]
        sample_name = list(input_df['sample'])[i]
        sample_bandgap = list(input_df['bandgap (eV)'])[i]
        sample_absorbtance = list(input_df['absorptance'])[i]
        sample_thickness = list(input_df['thickness (nm)'])[i]

        if measurements[i] == 'trpl':

            path = directory / list(input_df['trpl path'])[i]
            laser_intensity = list(input_df['trpl intensity'])[i]
            mode = list(input_df['trpl pulse type'])[i]

            exc_density = unpack_trpl_info(path, trpl_laser_reference_file, laser_intensity, sample_absorbtance, sample_thickness)
            time_processed, counts_normalized, noise_mask = import_trpl_data(path, mode, noise_threshhold)
            data[sample_index] = (time_processed, counts_normalized, noise_mask, exc_density)

        elif measurements[i] == 'plqe':

            path =  directory / list(input_df['plqe path'])[i]
            laser_wavelength = list(input_df['plqe laser'])[i]  # nm (wavelength)

            generation_rates, plqe  = import_plqe_data(path, sample_name, laser_wavelength, plqe_laser_reference_file, sample_absorbtance, sample_thickness)
            data[sample_index] = (generation_rates, plqe)

    return data