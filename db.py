import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import faiss
import math
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key="e6890c15-db55-4a6f-95e1-0810d25641bf")
model = SentenceTransformer('all-MiniLM-L6-v2')
from sklearn.metrics.pairwise import cosine_similarity

# creates sparse-dense embedding + query
# NOTE CAN ALSO GIVE DIFFERENT WEIGHTING TO EITHER EMBEDDING

# -----
INDEX = 'food-products'
MAX_CHUNK_SIZE = 700 # experiment with this

NAME_COL = 'name'
DESC_COL = 'info'
PRICE_COL = 'price'
ID_COL = 'id'
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

def save_index(database):
    index = pc.Index(INDEX)
    index.delete(delete_all=True, namespace='product-info')
    for _, row in tqdm(database.iterrows(), total=len(database)):
        vector = list(model.encode(row['entry']))
        vector = [float(v) for v in vector]
        index.upsert(
            vectors=[
                {"id": str(row['id']), "values": vector}
            ],
            namespace="product-info"
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
    save_index(database)

def query():
    index = pc.Index(INDEX)
    input_string = 'sweet chili flakes'
    print(f'query: {input_string}')
    input = list(model.encode(input_string))
    input = [float(v) for v in input]
    results = index.query(
        namespace='product-info',
        vector=input,
        top_k=3,
        include_values=False
    )
    return results, input_string

def get_cosine_score(id, input):
    data = pd.read_csv('planet_organic.csv')
    row = data.loc[data[ID_COL] == id]
    product_info = f'Name: {row[NAME_COL]}, price: {row[PRICE_COL]}, Info: {row[DESC_COL]}'
    p = model.encode(product_info)
    input = model.encode(input)
    score = cosine_similarity(p.reshape(1, -1), input.reshape(1, -1))
    return score[0][0]

if __name__ == '__main__':
    #add_data()
    results, input = query()
    chunk_ids = [entry['id'] for entry in results.matches]
    ids = [int(id)//10000 for id in chunk_ids]
    print(results)
    print(ids)
    for id in ids:
        print(get_cosine_score(id, input))
