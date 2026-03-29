import json
import ollama
import chromadb
import sys
import os.path

# Choose a sensible model
my_model=sys.argv[1] if len(sys.argv)==2  else "llama3:70b"
# run like this: python3 26-query-rag.py llama3.2:1b

queries="""
Are there any concerns I should have about the panel in terms of coverage?//
Are there any concerns I should have about the panel in terms of methods used for detecting metabolites?//
I am a computer scientist, I don't understand what LC-MS/MS and FIA-MS/MS are? What are the potential
strengths and weaknesses in terms of resolution. Please explain.//
Which metabolite classes in this panel are most relevant to systemic inflammation?//
Which metabolites in this panel would be expected to change with a structured 
exercise intervention?"//
If the dietary supplement contains omega-3 fatty acids, which panel classes should
show the largest changes at 6 months?"//
With 1233 metabolites across 49 classes, what pathway-level grouping strategy would you recommend 
             before running differential abundance analysis?"
"""

def embed(text):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return response["embedding"]

    
def query_panel(collection, question):
    query_embedding = embed(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5  # prob to big for example
    )
    context = "\n".join(results["documents"][0])
    response = ollama.chat(
        model=my_model,
        messages=[{
            "role": "user",
            "content": f"""You are a metabolomics expert.
            Context from the panel documentation:
            {context}
            Question: {question}"""
        }]
    )
    return response["message"]["content"]
            

chroma_db = "data/chroma_meta_db"
if not os.path.exists(chroma_db):
    sys.exit(f"Database {chroma_db} does not exist")
    

client = chromadb.PersistentClient(path=chroma_db)
collection = client.get_collection("metabolite_panel")

print("Starting questions")
for query in queries.split("//"):
    print("Query: ",query)
    print(query_panel(collection,query))
    
