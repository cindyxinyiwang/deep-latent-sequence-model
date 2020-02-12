"""download pretrained lm models from google drive
"""

import argparse
import requests
import tarfile
import subprocess
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data downloading")
    parser.add_argument('--dataset', choices=["yelp", "decipher", "sr_bos", "shakespeare", "all"],
        default="yelp", help='dataset to use')

    args = parser.parse_args()

    if not os.path.exists("data"):
        os.makedirs("data")

    os.chdir("pretrained_lm")

    yelp_id = ""
    decipher_id = "1GkfJ_bGATLQTq1xaDhHFE3ynQ1cYDNYX"
    sr_bos_id = "163KmzacA2QV7gFmp-o5pwHw7KSyhFX16"
    shakespeare_id = "1MRiysOHqcMHoGiPyYTcV5UlRpwNaRJcm"

    if args.dataset == "yelp":
        file_id = [yelp_id]
    elif args.dataset == "decipher":
        file_id = [decipher0_8_id]
    elif args.dataset == "sr_bos":
        file_id = [sr_bos_id]
    elif args.dataset == "shakespeare":
        file_id = [shakespeare_id]
    else:
        file_id = [yelp_id, decipher_id, sr_bos_id, shakespeare_id]

    destination = "datasets.tar.gz"

    for file_id_e in file_id:
        download_file_from_google_drive(file_id_e, destination)
        tar = tarfile.open(destination, "r:gz")
        tar.extractall()
        tar.close()
        os.remove(destination)
        # subprocess.run(["mv", "{}_style0"])

    os.chdir("../")
