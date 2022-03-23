# copied from https://github.com/pangyuteng/tcia-image-download-python
import zipfile
import sys,os
from tciaclient import TCIAClient
import pandas as pd
import traceback
import zipfile

def myunzip(zip_path,unzip_folder):
    os.makedirs(unzip_folder,exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_folder)


if __name__ == '__main__':
    
    csv_file_path = sys.argv[1]
    root_folder = sys.argv[2]

    if csv_file_path.endswith('.csv'):
        df = pd.read_csv(csv_file_path)

    # download images as zips
    #if "zip_file" not in df.columns:
    if not os.path.exists('downloaded.csv'):
        for n,row in df.iterrows():
            print(n,len(df))
            subject_id = row["Subject ID"]
            study_date = row["Study Date"]
            series_instance_uid = row["Series ID"]

            file_path = os.path.join(root_folder,subject_id,study_date,series_instance_uid,'img.zip')
            #if os.path.exists(file_path):
            #    continue
            
            os.makedirs(os.path.dirname(file_path),exist_ok=True)
            
            folder = os.path.dirname(file_path)
            basename = os.path.basename(file_path)

            tcia_client = TCIAClient(apiKey=None, baseUrl="https://services.cancerimagingarchive.net/services/v3",resource="TCIA")
            tcia_client.get_image(seriesInstanceUid=series_instance_uid,downloadPath=folder,zipFileName=basename)

            df.at[n,"zip_file"]=file_path

        df.to_csv(csv_file_path,index=False)

    # data_csv_file = "data.csv"
    # mylist = []
    # if not os.path.exists(data_csv_file):
    #     # raw no
    #     for n,row in df.iterrows():
    #         unzip_folder = os.path.dirname(row.zip_file)

    #         myunzip(row.zip_file,unzip_folder)
    #         #mylist.append('')
    #         break