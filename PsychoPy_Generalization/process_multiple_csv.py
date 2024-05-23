import os
import subprocess
from psychopy import gui
from data_preprocess_generalization import process_data_file

def process_multiple_csv_files():
    files = gui.fileOpenDlg()
    
    if not files:
        print("No files selected. Exiting.")
        return
    
    for csv_file_path in files:
        process_data_file(csv_file_path)

process_multiple_csv_files()

