import os
import pathlib
import glob
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS



# Directory containing your PDF folders
base_dir = r"C:\Users\Sreedev\Desktop\cv\temp"


# Using Pathlib and glob:
data_paths = []
for folder_path in glob.glob(os.path.join(base_dir, "*")):
    if os.path.isdir(folder_path):
        # Get all PDF files within the folder
        pdf_paths = glob.glob(os.path.join(folder_path, "*.pdf"))
        data_paths.extend(pdf_paths)

# Alternative using recursive globbing:
# data_paths = [str(p) for p in pathlib.Path(base_dir).rglob("*.pdf")]

print("Extracted PDF paths:")
if data_paths:
    for path in data_paths:
        print(path)
else:
    print("No PDF files found in the specified directory or its subdirectories.")



def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as pdf_file:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text.strip()
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return ""

# Extract text from all PDFs and store in a list
extracted_texts = []
for pdf_path in data_paths:
    extracted_texts.append(extract_text_from_pdf(pdf_path))


def strmaker(extracted_texts):
  str1=""
  for i in extracted_texts:
    for j in i:
      str1+=j

    return str1
docs=strmaker(extracted_texts)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
texts_chunks = text_splitter.split_text(docs)
documents = [Document(page_content=text) for text in texts_chunks]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = FAISS.from_documents(documents,embeddings)
print (vector_store)


vector_store.save_local("vector_store")#saving vectorstore locally