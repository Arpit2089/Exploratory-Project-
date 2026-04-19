import os
import urllib.request
import zipfile
import shutil

def download_and_extract_tiny_imagenet(base_dir='./data'):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_path = os.path.join(base_dir, 'tiny-imagenet-200.zip')
    extracted_folder = os.path.join(base_dir, 'tiny-imagenet-200')
    train_dir = os.path.join(extracted_folder, 'train')

    os.makedirs(base_dir, exist_ok=True)

    if not os.path.exists(train_dir):
        if not os.path.exists(zip_path):
            print("Downloading Tiny ImageNet (This may take a few minutes)...")
            # Robust chunked download
            with urllib.request.urlopen(url) as response, open(zip_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            print("Download complete.")
        
        print("Extracting zip file safely...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        print(f"Extraction complete! Validated train directory at: {train_dir}")
    else:
        print(f"Tiny ImageNet is already correctly installed at: {train_dir}")

if __name__ == '__main__':
    download_and_extract_tiny_imagenet()