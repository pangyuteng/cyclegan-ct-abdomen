import os
import sys
import glob
import pydicom
import pandas as pd


def main(root_folder,csv_file):
    mypath = os.path.join(root_folder,'**','*.dcm')
    dcm_file_list =[path for path in glob.glob(mypath,recursive=True)]
    print(len(dcm_file_list))
    mylist = []
    for dcm_file in dcm_file_list:
        ds = pydicom.dcmread(dcm_file,stop_before_pixels=True)        
        dcm_dict = {
            'series_description':ds.SeriesDescription,
            'dcm_file':dcm_file,
            'patient-id':ds.PatientID,
            'study_instance_uid':ds.StudyInstanceUID,
            'study_date':ds.StudyDate,
            'series_instance_uid':ds.SeriesInstanceUID,            
        }
        mylist.append(dcm_dict)

    pd.DataFrame(mylist).to_csv('data.csv',index=False)

if __name__ == "__main__":

    root_folder = sys.argv[1]
    csv_file = 'data.csv'
    main(root_folder,csv_file)

'''
cd prepare
python prepare.py /mydata/c4kc-kits
'''
