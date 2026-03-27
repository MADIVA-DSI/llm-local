# Running LLMs locally

Simple case study for MADIVA LLM course

## Exercise 1: Use of vLLM

This should work easily on a Linux machine with a GPU. For Linux  without a GPU or an Apple Silicon,
more gymnastics are need as specialist packages must be installed. For Apple it should be straightforward
but the instructions on the vLLM site failed for me.

I am usin using `pip3` but it is recommended to use frameworks such as `uv`. Python 3.12 is recommended but 3.10-3.13 should work. 

```
pip3 install vllm torch
```

We are now ready to run the exercise. Read through the code `01-vllm-driver.py` and run.

```
python3 01-vllm-driver.py
```

## Exercise 2: Use of Ollama

Our first exercise is a simple running of Ollama using the Python code.

If you weren't able to install ollama last week, try this

```
pip3 install ollama
```

(Again, recommended to use a package manager like `uv`)

Please read through the code carefully.

The simplest way to run is

```
python3 10-ollama.py
```

By default, the program uses `llama3:70b` as a model. If you are running on a CPU, choose a simpler model like `gemma3:12b` or `llama3.2:1b`.

```
python3 10-ollama.py llama3.2:1b
```

## Interlude

We will now see how RAG is done in three steps. For pedagogic reasons I am splitting into three steps but in
a real program you probably would do in one step or possibly two (steps 1 and 2; and step 3). Step 1 and 2 is about
building the database for RAG. This would typically be done once (or maybe a few times), where as step 3 and variants  and extensions may be done many times using the output of step 2.

What we'll be doing is make available knowledge of a particular metabolomics panel to our LLM through RAG. The input is  PDF downloaded from a vendor's web site. We'll do this in three steps

1. Extract the information from the PDF and store it in json format.
2. Embed the data in json format into a vector and store as a vector data base
3. Run queries using our LLM, augmenting through RAG with our vector database

## Exercise 3: Converting PDF to JSON

The first step doesn't have much to do with LLMs at all. It uses a python package called `pdfplumber` to extract out the information from the PDF. As it turns out this is the most complex  part of all the code. We could have done this very simply by effectively just converting PDF to text, but in many cases there may be value in using your higher-level knowledge of the document to generate more structured text. In this case, we have an overview of the panel, followed by the detail of the panel and use pdfplumber to get the data. Of course, I did not do this manually. I asked Clude to generate the code for me and then spent about 15 minutes tweaking it.

The program `21-extract.pdf` takes the PDF and parses it and stores the relevant information in a json file. Run the code and then look at the json output in the `data` directory.

```
python3 21-extract.pdf
```


## Exercise 4: Chunking and embedding

The next step is to take the json file as input and create a RAG database that can be queried by Ollama later. What this does it to chunk the file and embed the data in vectors. We use a library called chromadb for this purpose but there are other ways of doing this. In our code, we create a _persistent_ database because in our usage case we might run our query program that comes next several times and we don't want to go through the chunking exercise each time.

```
python3 24-chunk.py
```

The output goes into the `data` directory and is stored in a sqlite database. If you are feeling nosy you can use sqlite3 to poke about and look at the embeddings.

## Exercise 5. Querying

The third step is do querying. In a real scenario you probably would run many queries using the same database

```
python3 26-query-rag.py
```

By default, the program uses `llama3:70b` as a model. If you are running on a CPU, choose a simpler model like `gemma3:12b` or `llama3.2:1b`.

```
python3 26-rag-query.py llama3.2:1b
```







