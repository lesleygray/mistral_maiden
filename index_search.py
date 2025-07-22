import sys
import os 
import argparse
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Without this model.encode() hangs on mac
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

"""
This script builds and indexes OMOP vocabularies from Athena for semantic search or RAG. 
Script supports command line query and interactive search.
1. Applys some cleaning to Athena data 
2. Concatenates Athena concepts into string
3. Generates embeddings 
4. Indexes with euclidean distance 
5. Conducts similarity search (model.encode) and reports top 5 hits 

Usage CLI: 
  python index_search.py -i "blood pressure"
  
Usage Jupyter:
  Run cell 1 of transformer_search.ipynb
  Query: blood pressure

Limitations:
  - Interactive query has not been tested for Julpter - havent added widgets
  - Dies with >5000 concepts on M1, need to rebuild to enable threading (see line 87)

"""


# set inputs
input_concepts = "vocabularies/CONCEPT.csv"

# set outpus
output_concepts_clean = "vocabularies/CONCEPT_cleaned.csv"
output_index = "vocabularies/CONCEPT_cleaned.faiss"

# configurations
allowed_vocabularies = {"SNOMED", "LOINC", "RxNorm"}

# models
#embeddings_model = "BAAI/bge-base-en-v1.5" # accurate + fast + fails on M1
embeddings_model = "all-MiniLM-L6-v2"  # light + fast


#--------------------------------------------------------------
# check vars
input_paths = os.path.join(os.getcwd(), input_concepts)
#print(input_paths)

# Confirm a file exists
if os.path.isfile(input_paths):
    print("Environment looks OK")
else:
    print("Input paths are busted")


# Function to load concepts, filter data and writes concatenated string to CSV
# Filtering criteria: removes NAs from invalid_reason, keeps allowed_vocabularies and removes short concept_name
def clean_concepts(input_concepts, output_concepts_clean):
    print("Loading concept data...")
    df = pd.read_csv(input_concepts, dtype=str, sep="\t")
    #print(df.head())

    print("Cleaning...")
    df = df[df["invalid_reason"].isna()]
    df = df[df["vocabulary_id"].isin(allowed_vocabularies)]
    df = df[df["concept_name"].str.len() > 4]
    
    df["text"] = (
        df["concept_name"] + " | " +
        df["domain_id"] + " | " +
        df["vocabulary_id"] + " | concept_id: " + df["concept_id"]
    )
    print(df["text"].head())

    # Write to csv
    df.to_csv(output_concepts_clean, index=False)
    print(f"Cleaned data: {output_concepts_clean}")

	# debug - run first N concepts
	# Works well for 10,000 concepts then struggles (cant take 15,000)
    df = df[:5000]
    return df


# Function to convert concatenated context string to embeddings
def embed_text(df, embeddings_model):
    print(f"Embedding with model: {embeddings_model}")
    model = SentenceTransformer(embeddings_model)
    vectors = model.encode(df["text"].tolist(), show_progress_bar=True)
    return vectors, model

# Function to index vectors and save to file
def build_index(vectors, output_index):
    print("Building FAISS index...")
    dim = vectors[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    faiss.write_index(index, output_index)
    print(f"Index saved to {output_index}")
    return index

# Get top 5 hits
def interactive_search(index, df, model, top_k=5):
    print("\n Enter a search query (type 'exit' to quit):")

    while True:
        query = input("Query: ")
        if query.lower() in ["exit", "quit"]:
            break
        query_vector = model.encode([query])
        D, I = index.search(query_vector, k=top_k)
        print("\n Top Results:")
        for rank, idx in enumerate(I[0]):
            print(f"{rank+1}. {df.iloc[idx]['text']}")
        print("")


# Single query in Jupyter
def query_index(query_text, index, df, model, top_k=5):
    query_vector = model.encode([query_text])
    D, I = index.search(query_vector, k=top_k)
    print(f"\n Top {top_k} results for query: '{query_text}':\n")
    for rank, idx in enumerate(I[0]):
        print(f"{rank+1}. {df.iloc[idx]['text']}")


# Runs clean, embed, build and (optional) query
def clean_and_query(query=None):
    df = clean_concepts(input_concepts, output_concepts_clean)
    vectors, model = embed_text(df, embeddings_model)
    index = build_index(vectors, output_index)

    if query:
        query_index(query, index, df, model)
    else:
        interactive_search(index, df, model)

    return df, vectors, model, index

# For CLI 
def main(args=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('-v', '--verbose', dest="verbosity", action="count", default=0,
		help="How verbose logging should be. The more invocations, the more verbose (up to three)")
	parser.add_argument('-i', '--input', required=True, type=str, dest="query", default=None,
		help="A query string wrapped in quotes. For example: 'blood pressure'.")
	parameters = parser.parse_args(args)

	if parameters.query is None:
		parser.parse_args(['-h'])

	print("Running query: " + parameters.query)
	print("\n")
	
	clean_and_query(parameters.query)

# For CLI
if __name__ == "__main__": 
    main(sys.argv[1:])

