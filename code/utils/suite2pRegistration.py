import shutil
import os
import suite2p
import numpy as np
from tifffile import tifffile

def suite2pRegistration(data_dir, fn, n_time, Ly, Lx, output_path_temp, folder_number,output_path_suite2p):
    ops = np.load("../code/utils/ops.npy", allow_pickle = True).item()
    
    db = {
      # 'h5py': [], # a single h5 file path
      # 'h5py_key': 'data',
      'look_one_level_down': False, # whether to look in ALL subfolders when searching for tiffs
      'data_path': [os.path.join(data_dir, folder_number)], 
                    # a list of folders with tiffs 
                    # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)        
      # 'subfolders': [], # choose subfolders of 'data_path' to look in (optional)
    #   'fast_disk': 'C:/BIN', # string which specifies where the binary file will be stored (should be an SSD)
      'tiff_list': [os.path.basename(fn)], # list of tiffs in folder * data_path *!
      'save_path0': os.path.join(output_path_temp, folder_number) # TODO: make sure output_path is only str
    }

    opsEnd = suite2p.run_s2p(ops=ops, db=db)

    # Read in raw tif corresponding 
    # f_raw = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename=fname)
    # Create a binary file we will write our registered image to 
    f_reg = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename= opsEnd['save_path'] + '/data.bin', n_frames = n_time).data # Set registered binary file to have same n_frames

    # Delete the temporary file created by suite2p otherwise registration will not take place for the next trial

    # Delete the suite2p directory
    if os.path.exists(opsEnd["reg_file"]) and 'suite2p' in opsEnd["reg_file"]:
      print("Deleting the temporary suite2p directory.")
      shutil.rmtree(os.path.dirname(opsEnd["reg_file"]), ignore_errors=True)
    else:
      print(f"The directory {opsEnd['reg_file']} does not exist.")

    # with tifffile.TiffWriter(output_path_suite2, bigtiff=True) as tif:
    f_reg[f_reg < 0] = 0
    f_reg = np.uint16(f_reg)
    tifffile.imwrite(output_path_suite2p, f_reg)
    # f_reg.write_tiff(output_path_suite2p)

    x_shifts_mean = np.mean(opsEnd['xoff'])
    y_shifts_mean = np.mean(opsEnd['yoff'])

    # Center the shifts to zero 
    return opsEnd['xoff'] - x_shifts_mean, opsEnd['yoff'] - y_shifts_mean