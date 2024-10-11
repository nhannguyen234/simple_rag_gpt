import os
import glob

def get_documents_from_folder_path(folder_path):
    documents = []
    for path in glob.glob(os.path.join(folder_path,'*.txt')):
        documents.append(path)
    return documents