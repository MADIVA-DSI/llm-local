import ollama
import sys

# replace with llama3.2:1b on a laptop
model=sys.argv[1] if len(sys.argv)==2  else "llama3:70b"

queries="""
What is a metabalomics panel?//
What are the top two approaches for doing metabolomics?
"""

def query_panel(question):
    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": f"""You are a metabolomics expert. I am a computer scientist.
            Question: {question}"""
        }]
    )
    return response["message"]["content"]
            
print("Starting questions")
for query in queries.split("//"):
    print("Query: ",query)
    print(query_panel(query))
    
