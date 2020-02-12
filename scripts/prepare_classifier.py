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
    parser = argparse.ArgumentParser(description="classifer downloading")
    parser.add_argument('--dataset', choices=["yelp", "shakespeare", "all"],
        default="yelp", help='dataset to use')

    args = parser.parse_args()

    if not os.path.exists("pretrained_classifer"):
        os.makedirs("pretrained_classifer")

    os.chdir("pretrained_classifer")

    yelp_id = "1ZqjzxVx8ekOYTjNBYsI8FWT7UzHeZNOe"
    shakespeare_id = "16lsXqxgAS4V6naJiNYZ0VZ2ZofA81fEX"

    if args.dataset == "yelp":
        file_id = [yelp_id]
    elif args.dataset == "shakespeare":
        file_id = [shakespeare_id]
    else:
        file_id = [yelp_id, shakespeare_id]

    destination = "datasets.tar.gz"

    for file_id_e in file_id:
        download_file_from_google_drive(file_id_e, destination)
        tar = tarfile.open(destination, "r:gz")
        tar.extractall()
        tar.close()
        os.remove(destination)
        # subprocess.run(["mv", "{}_style0"])

    os.chdir("../")
