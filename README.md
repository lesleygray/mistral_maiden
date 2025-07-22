# Setup environment for mistral testing 
## 1. Load env
```bash
conda activate py312
```

## 2. Install packages
```bash
# Note pytorch requires GPU
conda install ipywidgets jupyterlab
conda install conda-forge::llm 
conda install markdown
conda install -c conda-forge transformers accelerate torch faiss-cpu datasets pandas
conda install transformers
conda install conda-forge::accelerate 
conda install pytorch::faiss-cpu
conda install conda-forge::datasets 
conda install conda-forge::pandas
conda install conda-forge::peft 
conda install conda-forge::uvicorn
conda install conda-forge::fastapi
conda install conda-forge::pydantic
conda install conda-forge::sqlalchem
which pip
pip install psycopg2
pip install sentence_transformers

# Full Mistral 7B too large to run on M1
export CMAKE_ARGS="-DLLAMA_METAL=on"
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## 3. Download a model 
Go to [Hugging Face] (https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf) and download quantized mistral 7B.
Save to working location: '/Users/lelgray/Dropbox/MCRI/mistral_testing/'

## 4. Test model install
```python

'''
from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
	filename="mistral-7b-instruct-v0.2.Q2_K.gguf",
)

llm.create_chat_completion(
	messages = [
		{
			"role": "user",
			"content": "What is the capital of France?"
		}
	]
)
'''

from llama_cpp import Llama

llm = Llama(model_path="models/mistral-7b-instruct.Q4_K_M.gguf", n_ctx=2048)

output = llm("### Instruction:\nMap 'Patient Gender' to OMOP.\n\n### Response:\n", max_tokens=256)
print(output["choices"][0]["text"])
```

## 5. Download OHDSI vocabularies from Athena

Go to [Athena OHDSI](https://athena.ohdsi.org/) and download the following vocabularies (same as used by Xenti):

- **RxNorm**
- **LOINC**
- **SNOMED**

Extract files and store somewhere 

## 6a. Load with Pandas (Python)
```python
import pandas as pd

df = pd.read_csv("vocabularies/CONCEPT.csv", dtype=str)
```

## 6b. Load into PostgreSQL (higher performance)
```bash
psql -d your_db -c "\COPY concept FROM 'athena_download/CONCEPT.csv' CSV HEADER"
```
