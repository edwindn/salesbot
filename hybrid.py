import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import faiss
import math
import nltk
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
pc = Pinecone(api_key="e6890c15-db55-4a6f-95e1-0810d25641bf")
model = SentenceTransformer('all-MiniLM-L6-v2')
sparse_encoder = BM25Encoder()

nltk.download('punkt_tab')

# this script creates the FAISS index that faiss_search can use to return the n most relevant products for a given query

# -----
INDEX = 'food-products-hybrid'
QUERY = 'tomatoes to go with pasta'
MAX_CHUNK_SIZE = 700 # experiment with this

NAME_COL = 'name'
DESC_COL = 'info'
PRICE_COL = 'price'
ID_COL = 'id'
URL_COL = 'url'
SPECS_COL = None # important data to inclue in every chunk
# -----

def get_search_database(data): # assuming we have the columns in the format given above
    search_database = pd.DataFrame(columns=['id', 'entry'])
    for _, row in data.iterrows():
        master_id = row[ID_COL]

        try:
            description = row[DESC_COL].strip().replace('\n\n' ,'\n').replace('\n', ' ')
        except:
            continue
        description = recursive_split(description)

        for i, chunk in enumerate(description):
            if SPECS_COL:
                entry = {
                    'id': int(str(master_id) + f'{i+1:04d}'), # last 4 digits are chunk-specific (ignorable in overall search)
                    'entry': f"Product name: {row[NAME_COL]}. Price: {row[PRICE_COL]}\n{chunk}\n{row[SPECS_COL]}"
                }
            else:
                entry = {
                    'id': int(str(master_id) + f'{i+1:04d}'),
                    'entry': f"Product name: {row[NAME_COL]}. Price: {row[PRICE_COL]}\n{chunk}"
                }
        
            search_database.loc[len(search_database)] = entry

    return search_database

def fit_sparse_encoder(data):
    corpus = []
    for _, row in data.iterrows():
        try:
            entry = row[DESC_COL].strip().replace('\n\n' ,'\n').replace('\n', ' ')
        except:
            continue
        corpus.append(entry)
    sparse_encoder.fit(corpus)

# train with sparse_encoder.fit([array of strings]) to train on the entire corpus
# encode with sparse_encoder.encode_documents(string)

def save_index(database, reset=False):
    index = pc.Index(INDEX)
    if reset:
        index.delete(delete_all=True)
    for _, row in tqdm(database.iterrows(), total=len(database)):
        vector = list(model.encode(row['entry']))
        vector = [float(v) for v in vector]
        sparse_vector = sparse_encoder.encode_documents(row['entry'])

        index.upsert(
            vectors=[
                {
                    "id": str(row['id']),
                    "values": vector,
                    "sparse_values": sparse_vector
                }
            ],
            #namespace="product-info"
        )

    print(f'Upserted all data to {INDEX}')

def recursive_split(text, max_chunk_size=MAX_CHUNK_SIZE):
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = split_by_sentence(text, max_chunk_size)

    return chunks

def split_by_sentence(text, max_chunk_size):
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    chunks = reduce_chunks(chunks, max_chunk_size)
    return chunks

def reduce_chunks(chunks, max_chunk_size):
    new_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_size:
            chunk1, chunk2 = split_in_half(chunk)
            new_chunks.extend(reduce_chunks([chunk1, chunk2], max_chunk_size))
        else:
            new_chunks.append(chunk)
    return new_chunks

def split_in_half(chunk):
    idx = len(chunk) // 2
    split_idx = chunk.rfind(' ', 0, idx)
    if split_idx == -1:
        split_idx = idx
    return chunk[:split_idx].strip(), chunk[split_idx:].strip()

# -----
def add_data():
    data = pd.read_csv('planet_organic.csv')
    database = get_search_database(data)
    database.to_csv('search_database.csv')
    save_index(database, reset=False)

def query():
    index = pc.Index(INDEX)
    input = QUERY
    dense_input = list(model.encode(input))
    dense_input = [float(v) for v in dense_input]
    sparse_input = sparse_encoder.encode_documents(input)
    results = index.query(
        #namespace='product-info',
        vector=dense_input,
        sparse_vector=sparse_input,
        top_k=5,
        include_values=False
    )
    chunk_ids = [entry['id'] for entry in results.matches]
    product_ids = [int(id)//10000 for id in chunk_ids]
    return product_ids
    return results


def get_info(id):
    data = pd.read_csv('planet_organic.csv')
    row = data.loc[data[ID_COL] == id]
    row = f'NAME: {row[NAME_COL].values[0]}, PRICE: {row[PRICE_COL].values[0]}, URL: {row[URL_COL].values[0]}\nPRODUCT INFO: {row[DESC_COL].values[0]}'
    return row

def main():
    data = pd.read_csv('planet_organic.csv')
    fit_sparse_encoder(data)
    #add_data()
    results = get_info(170)
    print(results)
    return
    ids = [(entry['id']) for entry in results.matches]
    print([int(id)//10000 for id in ids])
    return
    chunk_ids = [entry['id'] for entry in results.matches]
    ids = [int(id)//10000 for id in chunk_ids]
    print(chunk_ids)
    print(ids)

if __name__ == '__main__':
    main()