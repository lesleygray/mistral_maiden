import os
import json
import pandas as pd
import faiss
import difflib
import re
import streamlit as st
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

"""
Author: Lesley Gray
Date: 21/07/2025

This script does:
    1. Maps user-defined fields to OMOP concepts using embeddings + LLM
    2. Uses FAISS for similarity search, llama.cpp for inference
    3. Streamlit app for simple input/output UI

Run:
    conda activate py312
    streamlit run mistral_streamlit.py

Limitations:
 - pip install of llama-python-cpp=0.3.14 throws seg fault with v0.2_Q5_K_M
   - was due to threading, have tested with 0.2.24 and runs fine
 - Outputs have not been verified and are sometimes a bit garbage - json processing code needs review
    
"""

# -------------------------------
# CONFIG
concepts_clean = "vocabularies/CONCEPT_cleaned.csv"
concepts_clean_index = "vocabularies/CONCEPT_cleaned.faiss"
embeddings_model = "all-MiniLM-L6-v2"
llm_model = "models/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
#llm_model = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP (used by BLAS, llama.cpp)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # HuggingFace tokenizers
os.environ["LLAMA_CPP_THREADS"] = "1"  # Llama-cpp specific (optional)
query_history = []


# -------------------------------

# Globals (set in init_models)
concept_df = None
faiss_index = None
embeddings = None
llm = None

# Load data
def load_concepts():
    input_paths = os.path.join(os.getcwd(), llm_model)
    if not os.path.isfile(input_paths):
        st.error(f"Model file not found: {input_paths}")
        st.stop()
    df = pd.read_csv(concepts_clean, dtype=str, sep=",")
    index = faiss.read_index(concepts_clean_index)
    return df, index

# Load models
@st.cache_resource
def init_models():
    global concept_df, faiss_index, embeddings, llm
    concept_df, faiss_index = load_concepts()
    embeddings = SentenceTransformer(embeddings_model)
    print("Workdir is:" + os.getcwd())
    print("Model path is:" + llm_model)
    # Unable to set n_ctx in v0.2.24 - llama_cpp.Llama doesnt initialise 
    # Runs with seg fault
    llm = Llama(model_path=llm_model) #, n_ctx=256, n_gpu_layers=1)
    return concept_df, faiss_index, embeddings, llm

# Get top OMOP matches
def get_top_omop_concepts(query, k=5):
    query_vec = embeddings.encode([query])
    D, I = faiss_index.search(query_vec, k)
    #concepts = [concept_df.iloc[i]["text"] for i in I[0]]

    # debug
    # print("Indexes returned from FAISS:", I)
    # print("Concept DataFrame shape:", concept_df.shape)
    # print("Concept DataFrame columns:", concept_df.columns)
    concepts = concept_df.iloc[I[0]]["text"].tolist()
    #concepts = [
    #    concept_df.iloc[i]["text"]
    #    for i in I[0]
    #    if 0 <= i < len(concept_df) and "text" in concept_df.columns
    #]
    return "\n".join([f"{i+1}. {c}" for i, c in enumerate(concepts)])

# Run prompt through LLM
def run_prompt(prompt):
    result = llm(prompt, max_tokens=512)
    return result["choices"][0]["text"]

# Improve output with concept match
def find_best_concept_match(output_text):
    print("OK22")
    print(output_text)
    print("OK33")
    try:
        # Throws message of malformatted json 
        #parsed = json.loads(output_text.split("```")[-1])
        try:
            json_str = re.search(r"\{.*\}", output_text, re.DOTALL).group()
            parsed = json.loads(json_str)
            print("json_string is:" + json_str)
            print("parsed is:" + parsed)
        except (AttributeError, json.JSONDecodeError) as e:
            st.error(f"Could not parse JSON from output: {e}")
            st.write("Raw output was:", output_text)
        
        name = parsed.get("omop_target_field", "")
        if name:
            best = difflib.get_close_matches(name.lower(), concept_df["concept_name"].str.lower(), n=1)
            if best:
                match = concept_df[concept_df["concept_name"].str.lower() == best[0]].iloc[0]
                parsed["omop_concept_id"] = match["concept_id"]
                parsed["match_score"] = 1.0
                return json.dumps(parsed, indent=2)
    except Exception as e:
        st.warning(f"Matching error: {e}")
    return output_text

# Streamlit UI
def main():    
    # clear history
    with open("history.json", "w") as io:
        pass

    global concept_df, faiss_index, embeddings, llm
    concept_df, faiss_index, embeddings, llm = init_models()
    st.set_page_config(page_title="OMOP Mapping Tool", layout="wide")
    st.title("OMOP Concept Mapper")
    
    # Load models (runs once)
    with st.spinner("Loading models..."):
        st.success("Models loaded.")

    # Input
    field_query = st.text_area("Enter a field or variable to map:", "Patient Gender", height=100)

    if st.button("Submit to Chat"):
        st.info("Processing your request...")

        omop_context = get_top_omop_concepts(field_query)
        q = f"""
            You are an AI trained to map clinical data dictionaries to OMOP CDM.

            Here are the most relevant OMOP concepts for '{field_query}':{omop_context}
            Field: {field_query}
            Description: {field_query}

            Please return this JSON format:
                {{
                "omop_target_table": "",
                "omop_target_field": "",
                "omop_concept_id": "",
                "data_type_mapping": "",
                "pii_detection": false,
                "consent_detection": false,
                "consent_mapping": "",
                "confidence_score": 0.85
            }}
        """
        
        response = run_prompt(q)
        #response = '{"omop_target_table": "", "omop_target_field": "", "omop_concept_id": "", "data_type_mapping": "", "pii_detection": false, "consent_detection": false, "consent_mapping": "", "confidence_score": 0.85}'
        print("Response is: " + response)
        print("OK11")
        enhanced_output = find_best_concept_match(response)

        # Log history
        with open("history.json", "a") as io:
            json.dump({"q": q, "response": enhanced_output}, io)
            io.write("\n")

        st.subheader("LLM Output with Concept Match")
        st.code(enhanced_output, language="json")

if __name__ == "__main__":
    main()
