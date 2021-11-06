import zipfile

pre_dir = './content/archive.zip'
target_dir = './content/dataset/cnn/data'

zf = zipfile.ZipFile(pre_dir)
zf.extractall(target_dir)