import json
import ollama
import chromadb


queries="""
Are there any concerns I should have about the panel?//
I am a computer scientist, I don't understand what LC-MS/MS and FIA-MS/MS are what the potential
strengths and weaknesses are. Please explain.//
Which metabolite classes in this panel are most relevant to systemic inflammation?//
Which metabolites in this panel would be expected to change with a structured 
exercise intervention?"//
If the dietary supplement contains omega-3 fatty acids, which panel classes should
show the largest changes at 6 months?"//
With 1233 metabolites across 49 classes, what pathway-level grouping strategy would you recommend 
             before running differential abundance analysis?"
"""


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
   client = chromadb.Client()
   collection = client.create_collection("metabolite_panel")

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
    
def query_panel(collection, question):
    query_embedding = embed(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    context = "\n".join(results["documents"][0])
    response = ollama.chat(
        model="llama3:70b",
        messages=[{
            "role": "user",
            "content": f"""You are a metabolomics expert.
            Context from the panel documentation:
            {context}
            Question: {question}"""
        }]
    )
    return response["message"]["content"]
            
metabolite_records=json.load(open("data/metabolites.json"))

print("Building chunks")
chunks=build_chunks(metabolite_records)
print("Starting embedding")
collection=embed_chunks(chunks)
print("Staring questions")
for query in queries.split("//"):
    print("Query: ",query)
    print(query_panel(collection,query))
    
