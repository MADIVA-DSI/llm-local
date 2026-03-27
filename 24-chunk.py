import json
import chromadb
import ollama




def build_chunks(metabolite_records):
    chunks = []
    # Group by class
    from itertools import groupby
    for class_name, members in groupby(metabolite_records, key=lambda x: x['class']):
        members = list(members)
        names = [m['full_name'] for m in members]
        chunk = f"""
      Metabolite class: {class_name}
      Number of analytes: {len(members)}
      Analytical method: {members[0]['method']}
      Metabolites: {', '.join(names)}
"""
        chunks.append({"class": class_name, "text": chunk, "count": len(members)})
    return chunks


# Embed chunks using Ollama's local embedding model
def embed(text):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return response["embedding"]

def embed_chunks(chunks):
   client = chromadb.PersistentClient(path="data/chroma_meta_db")
   collection = client.get_or_create_collection("metabolite_panel")
   for i, chunk in enumerate(chunks):
       print("Chunk ",i)
       collection.add(
           ids=[str(i)],
           embeddings=[embed(chunk["text"])],
           documents=[chunk["text"]],
            metadatas=[{                         # optional but useful for filtering
                "class":  chunk["class"],
                "count":  str(chunk["count"]),
            }]
       )
   return collection
    
            
metabolite_records=json.load(open("data/metabolites.json"))

print("Building chunks")
chunks=build_chunks(metabolite_records)
print("Starting embedding")
collection=embed_chunks(chunks)

# Database is persisted automatically so nothing to do

