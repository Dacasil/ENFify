import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
sys.path.insert(0, os.path.abspath('../sources'))

try:
    from ENF_Enhancement import VariationalModeDecomposition
except ImportError as e:
    print(f"Import Error: {e}")

try: 
    from PDF_and_Plot import cut_to_alpha_pdf, to_alpha_pdf, read_wavfile, create_phase_plot, create_cut_phase_plot
except ImportError as e:
    print(f"Import Error: {e}")

try: 
    from ENF_preprocessing import downsampling, bandpass_filter
except ImportError as e:
    print(f"Import Error: {e}")

try: 
    from ENF_frequency_phase_estimation import segmented_phase_estimation_DFT0, segmented_phase_estimation_hilbert 
except ImportError as e:
    print(f"Import Error: {e}")

try: 
    from Rodriguez_Audio_Authenticity import find_cut_in_phases
except ImportError as e:
    print(f"Import Error: {e}")


###______________________________Main Code______________________________###


def main():
    # Create the parser for the CLI
    parser = argparse.ArgumentParser(
        description="ENFify - Audio Tampering Detection Tool"
    )
    # Add arguments
    parser.add_argument(
        'Audio_file_name',
        type=str,
        help="The name of the audio file to process."
    )
    parser.add_argument(
        '--downsampling',
        action='store_true',
        help="Enable downsampling of the audio file."
    )
    parser.add_argument(
        '--bandpassfilter',
        action='store_true',
        help="Enable bandpass filtering on the audio file."
    )
    parser.add_argument(
        '--VMD',
        action='store_true',
        help="Enable Variational Mode Decomposition."
    )
    # Parse the arguments
    args = parser.parse_args()
    
    # Access the arguments
    print(f"Processing file: {args.Audio_file_name}")
    if args.downsampling:
        print("Downsampling is enabled")
    if args.bandpassfilter:
        print("Bandpassfilter is enabled")
    if args.VMD:
        print("Variational Mode Decomposition is enabled")
    

    ###..................Read in the raw data..................###
    sig, fs = read_wavfile(f"INPUT_Audio_Data/{args.Audio_file_name}")
    

    ###...................Data Preprocessing...................###

    # Downsampling
    if args.downsampling == True:
        
        # Select the downsample frequency
        downsample_freq = float(input("Set the downsample frequency: "))

        # File paths to get the raw and save the downsampled data
        input_file = os.path.join("INPUT_Audio_Data",args.Audio_file_name)
        output_file = os.path.join("INPUT_Audio_Data/downsampled","downsampled_"+ args.Audio_file_name)

        # Downsample the signal data to 1000 Hz for lighter numeric calculations 
        downsampling(input_file, output_file, downsample_freq)
        sig, fs = read_wavfile(output_file)

        # Remove the downsampled data to enable programm 
        os.remove(output_file)


    # Bandpass Filter
    if args.bandpassfilter == True: 
        
        # Select the bandpass frequencies
        lowcut = float(input("Set the bandpass minimum frequency: "))
        high_cut = float(input("Set the bandpass maximum frequency: "))

        # Apply bandpass filter on the signal
        sig = bandpass_filter(sig, lowcut, high_cut, fs, 1)


    # Variational Mode Decomposition
    if args.VMD == True:
        vmd_settings = input("Continue with default VMD settings press \x1B[0m\x1B[3m Enter \x1B[0m  or type \x1B[3m True \x1B[0m to set new VMD settings:")

        if vmd_settings == True:
            alpha = float(input("Bandwith constraint alpha:"))
            tau = float(input("Noise-tolerance tau:"))
            n_mode = float(input("Number of modes:"))
            DC = str(input("Impose DC part"))
            tol = str(input("Error tolerance:"))

            u_clean,_,_ = VariationalModeDecomposition(sig, alpha, tau, n_mode, DC, tol)
            sig = u_clean[0]
        
        else:
            alpha = 5000 # moderate bandwidth constraint
            tau = 0 # noise-tolerance
            n_mode = 1 # Number of espected modes in the signal 
            DC = 0  # no DC part imposed  
            tol = 1e-7
            u_clean,_,_ = VariationalModeDecomposition(sig, alpha, tau, n_mode, DC, tol)
            sig = u_clean[0]    

    # Robust filtering algorithm (In progress)


    ###.....................Phase analysis.....................###

    # Set the Constants
    ref_enf = float(input("Expected ENF in Hz"))
    NUM_CYCLES = 10
    N_DFT = 20_000
    time = len(sig)/fs # time

    # Hilbert instantaneous phase estimateion
    hilbert_phases = segmented_phase_estimation_hilbert(sig, fs, NUM_CYCLES, ref_enf)
    x_hilbert = np.linspace(0., time, len(hilbert_phases))

    # DFT0 instantaneous phase estimation
    DFT0_phases = segmented_phase_estimation_DFT0(sig, fs, NUM_CYCLES, N_DFT, ref_enf)
    x_DFT0 = np.linspace(0.,time, len(DFT0_phases))

    hilbert_phases_new, x_hilbert_new, hil_interest_region = find_cut_in_phases(hilbert_phases, x_hilbert)
    DFT0_phases_new, x_DFT0_new, DFT0_interest_region = find_cut_in_phases(DFT0_phases, x_DFT0)

    image_path = "OUTPUT_Audio_Data"
    hilbert_phase_im = "hilbert_phase_im.png"
    DFT0_phase_im = "DFT0_phase_im.png"

    if hil_interest_region == 0:
        create_phase_plot(x_hilbert, hilbert_phases, image_path, hilbert_phase_im)
        create_phase_plot(x_DFT0, DFT0_phases, image_path, DFT0_phase_im)
        to_alpha_pdf(image_path + '/' + hilbert_phase_im, image_path + '/' + DFT0_phase_im)
    
    
    create_cut_phase_plot(x_hilbert_new, hilbert_phases_new, x_hilbert, hil_interest_region, image_path, hilbert_phase_im)
    create_cut_phase_plot(x_DFT0_new, DFT0_phases_new, x_DFT0, DFT0_interest_region, image_path, DFT0_phase_im)
    
    cut_to_alpha_pdf(image_path + '/' + hilbert_phase_im, image_path + '/' + DFT0_phase_im)

if __name__ == "__main__":
    main()


