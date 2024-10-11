import os
import sys

# Get the current file's directory (subfolder2)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up two directories to the 'src' folder
src_root = os.path.abspath(os.path.join(current_dir, "../../../"))

# Add 'src' folder to sys.path
sys.path.insert(0, src_root)

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from src.core.config import configs
from src.utils.helper_functions import get_documents_from_folder_path
from uuid import uuid4


embeddings = OpenAIEmbeddings(api_key=configs.OPENAI_API_KEY,
                              model='text-embedding-3-small')

# load documents
documents_path = get_documents_from_folder_path('./data')
documents = [TextLoader(path).load() for path in documents_path]

# Chunking documents
text_splitter = RecursiveCharacterTextSplitter(
    separators=['\n\n','\n'],
    chunk_size = configs.CHUNK_SIZE,
    chunk_overlap = 0
)
texts = [Document(
            page_content = text.page_content,
            metadata = text.metadata,
            id = (id_doc+1)*(id_text+1))
      for id_doc, doc in enumerate(documents)
        for id_text, text in enumerate(text_splitter.split_documents(doc))]
print(f'Number of chunking: {len(texts)}')

# set id for each chunk
uuids = [str(uuid4()) for _ in range(len(texts))]

# Create vectorstore in Chroma
vector_store = Chroma.from_documents(
    collection_name='illuminus_collection',
    documents=texts,
    embedding=embeddings,
    ids=uuids,
    persist_directory='./chroma_langchain_db' # save vectorstore locally
)
vector_store.persist()



