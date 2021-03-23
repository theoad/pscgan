from __future__ import print_function
import pickle
import os
import io
import argparse
import json
import os.path
import sys
from tqdm import tqdm
from utils.utils import mkdir
from utils.argparse_utils import PossibleDatasets, PossibleMethods
from googleapiclient.discovery import build, MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials

# If modifying these scopes, delete the file token.pickle.
SCOPES = {
    "rw": ['https://www.googleapis.com/auth/drive'],
    "ro": ['https://www.googleapis.com/auth/drive.readonly']
}

with open('utils/checkpoint_ids.json') as f:
    checkpoint_ids = json.load(f)


def build_service(mode="ro"):
    """
    taken from https://developers.google.com/drive/api/v3/quickstart/python
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is created automatically when
    # the authorization flow completes for the first time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if os.path.exists('token.json'):
        with open('token.json', 'rb') as token:
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES[mode])
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)
    return service


def upload_folder_helper(service, folder_name, parent_path, parent_id=None, recurse=True, verbose=True):
    """
    Uploads the given folder and its content to the provided service
    Return IDs, a dictionary containing all the ids of the uploaded files.
    """
    ids = {}
    if parent_id is not None:
        file_metadata = {
            'name': folder_name,
            'parents': [parent_id],
            'mimeType': 'application/vnd.google-apps.folder'
        }
    else:
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
    file = service.files().create(body=file_metadata, fields='id').execute()
    folder_id = file.get('id')
    for file in os.listdir(os.path.join(parent_path, folder_name)):
        if os.path.isdir(os.path.join(parent_path, folder_name, file)):
            if recurse:
                ids[file] = upload_folder_helper(service, file, os.path.join(parent_path, folder_name),
                                                 parent_id=folder_id, recurse=recurse, verbose=verbose)
            else:
                continue
        else:
            file_metadata = {
                'name': file,
                'parents': [folder_id]
            }
            file_path = os.path.join(parent_path, folder_name, file)
            media = MediaFileUpload(file_path, resumable=True)
            file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            ids[file_path] = file.get('id')
            if verbose:
                print(f"uploaded {file_path}, id = {ids[file_path]}")
    return ids


def upload_folder(service, folder_path, recurse=True, verbose=True):
    return upload_folder_helper(service,
                                os.path.basename(folder_path),
                                os.path.dirname(folder_path),
                                parent_id=None,
                                recurse=recurse,
                                verbose=verbose)


def download_file(service, path_to_save, file_id, verbose=True):
    if verbose:
        print(f"downloading {path_to_save}, id = {file_id}")
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    pbar = None
    if verbose:
        pbar = tqdm(total=100)
    while done is False:
        status, done = downloader.next_chunk()
        if verbose:
            pbar.update(int(status.progress() * 100))
    if verbose:
        pbar.close()

    with open(path_to_save, "wb") as f:
        f.write(fh.getbuffer())


def download_checkpoint(service, dataset, noise_std, method, dest_path, verbose=True):
    target_id = checkpoint_ids[dataset][f'noise{noise_std}'][f'{method}-{noise_std}.ckpt']
    download_file(service, os.path.join(dest_path, f'{dataset}-{method}-{noise_std}.ckpt'), target_id, verbose=verbose)


def download_all(service, dest_path, verbose=False):
    for dataset in checkpoint_ids:
        for noise in checkpoint_ids[dataset]:
            for ckpt in checkpoint_ids[dataset][noise]:
                path_to_save = os.path.join(dest_path, dataset, noise)
                mkdir(path_to_save)
                target_id = checkpoint_ids[dataset][noise][ckpt]
                download_file(service, os.path.join(path_to_save, ckpt), target_id, verbose=verbose)


def add_download_args(parent) -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(parents=[parent])
    arg_parser.add_argument("--all", help="If set, download all checkpoints", action='store_true')
    arg_parser.add_argument("--init", help="If set, tries to enable connection with PyDrive, and downloads token.pickle",
                            action='store_true')
    arg_parser.add_argument("--dataset", help="Which dataset to download.",
                            type=PossibleDatasets, choices=list(PossibleDatasets),
                            required="--all" not in sys.argv and '--init' not in sys.argv)
    arg_parser.add_argument("--noise_std", help="the standard deviation of additive white Gaussian noise contamination.",
                            type=int, required="--all" not in sys.argv and '--init' not in sys.argv)
    arg_parser.add_argument("--method", help="denoising algorithm to download.",
                            type=PossibleMethods, choices=list(PossibleMethods),
                            required="--all" not in sys.argv and '--init' not in sys.argv)
    arg_parser.add_argument("--out_dir", help="The directory where the checkpoints will be saved", type=str, default=".")
    arg_parser.add_argument("--silent", help="set to disable verbose", action='store_true')
    return arg_parser
