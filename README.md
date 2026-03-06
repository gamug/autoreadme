# Autoreadme
## Description
This development is dessign to auto-readme generation. We based the functions in [readme_ready](https://github.com/souradipp76/ReadMeReady/) development re-structuring and droping some sections to maintain basic functionality focused in comertial LLMs instead of fine tuned ones.

## Requirements
Main requiremens include langchain, transformers and python-magic (with binary files). For more details see ```requirements.txt```

### Environment setup
To setup the environment run

```bash
conda create -n readme python==3.10
conda activate readme
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Environment variables
*HF_TOKEN:* HuggingFace token for transformers SDK loging
*LLM_API_KEY:* Azure OpenAI API key to consume LLM
*LLM_API_URI:* URL to perform requests to the model
*LLM_DEPLOYMENT:* Azure OpenAI deployment name to use as LLM source.
*LLM_API_VERSION:* Azure OpenAI API version.

## Usage
The code executes a chain of steps to achieve its goal:
- **STEP 1; Understanding files and folders:** With the LLM help, generates a json per file and ```summary.json``` per folder with the understanding of the code.
- **STEP 2; Vectordb creation:** Uses ```hnswlib``` and an embeddings mode (depending on the user election) to structure a non-SQL database for further RAG usage.
- **STEP 3; Structuring readme file:** Use LLM to structuring section by section of the readme file.

This development adaptation is dessigning to run with minimal inputs. We has a common example in ```create_readme.py``` file with the following structure:
### LLM selection using available models. 
This is made by using ```LLModels``` class. Correct way to use the the class is as follows

```python
from autoreadme.types import LLMModels
llm = LLMModels.GPT5
```
Available models are:
- GPT5 (str): OpenAI GPT-5 model.
- GPT52 (str): OpenAI GPT-5.2 model.
- GPT3 (str): OpenAI GPT-3.5-turbo model.
- GPT4 (str): OpenAI GPT-4 model.
- GPT432k (str): OpenAI GPT-4-32k model with extended context window.
- TINYLLAMA_1p1B_CHAT_GGUF (str): TinyLlama 1.1B Chat model from
    TheBloke with GGUF format.
- GOOGLE_GEMMA_2B_INSTRUCT_GGUF (str): Gemma 2B Instruction model
    in GGUF format by bartowski.
- LLAMA2_7B_CHAT_GPTQ (str): LLaMA 2 7B Chat model using GPTQ
    from TheBloke.
- LLAMA2_13B_CHAT_GPTQ (str): LLaMA 2 13B Chat model using GPTQ
    from TheBloke.
- CODELLAMA_7B_INSTRUCT_GPTQ (str): CodeLlama 7B Instruction model
    using GPTQ from TheBloke.
- CODELLAMA_13B_INSTRUCT_GPTQ (str): CodeLlama 13B Instruction model
    using GPTQ from TheBloke.
- LLAMA2_7B_CHAT_HF (str): LLaMA 2 7B Chat model hosted on
    Hugging Face.
- LLAMA2_13B_CHAT_HF (str): LLaMA 2 13B Chat model hosted on
    Hugging Face.
- CODELLAMA_7B_INSTRUCT_HF (str): CodeLlama 7B Instruction model
    hosted on Hugging Face.
- CODELLAMA_13B_INSTRUCT_HF (str): CodeLlama 13B Instruction model
    hosted on Hugging Face.
- GOOGLE_GEMMA_2B_INSTRUCT (str): Gemma 2B Instruction model by Google.
- GOOGLE_GEMMA_7B_INSTRUCT (str): Gemma 7B Instruction model by Google.
- GOOGLE_CODEGEMMA_2B (str): CodeGemma 2B model by Google for
    code-related tasks.
- GOOGLE_CODEGEMMA_7B_INSTRUCT (str): CodeGemma 7B Instruction
    model by Google.

### Repository setup
To achieve this we dispose of ```AutodocRepoConfig``` class. You should use the object as follows:

```python
from autoreadme.types import LLMModels, Priority
from autoreadme.types import AutodocRepoConfig
repo_config = AutodocRepoConfig (
    name = "name_of_your_development", 
    root = "folder_in_your_laptop_where_development_is_located",
    repository_url = "github_link_to_main_branch_of_your_development",
    output = "folder_in_your_laptop_to_place_auto_generated_readme_file",
    llms = list[LLMModels],  #eg [LLMModels.GPT5, LLMModels.GPT52]
    ignore = [
        ".*",
        "*package-lock.json",
        "*package.json",
        "node_modules",
        "*dist*",
        "*build*",
        "*test*",
        "*.svg",
        "*.md",
        "*.mdx",
        "*.toml",
        "*__pycache__*",
        "*__init__.py",
        ".pytest_cache*",
        ".vscode*",
    ],  #files and folders to ignore in readme generation
    file_prompt = """Write a detailed technical explanation of
            what this code does. Focus on the high-level
            purpose of the code and how it may be used in the
            larger project.\nInclude code examples where
            appropriate. Keep you response between 100 and 300
            words. DO NOT RETURN MORE THAN 300 WORDS.
            Output should be in markdown format.
            Do not just list the methods and classes in this file.""",  #prompt to complement file llm understanding
    peft_model_path=None,
    folder_prompt = """Write a technical explanation of what the
            code in this file does and how it might fit into the
            larger project or work with other parts of the project.
            Give examples of how this code might be used. Include code
            examples where appropriate. Be concise. Include any
            information that may be relevant to a developer who is
            curious about this code. Keep you response under
            400 words. Output should be in markdown format.
            Do not just list the files and folders in this folder.""",  #prompt to complement folder llm understanding
    chat_prompt = None,  # Aditional instructions to take into account on readme generation. Can be string, None or bool
    content_type = "python code",  #Type of content LLM should expect in the files to analyse
    target_audience = "smart developer",  #Audience to whom the document is addressed
    link_hosted = True,  #Whether to generate hosted links in the documentation.
    priority = Priority.PERFORMANCE,  #Priority in the LLM usage. Could be COST or PERFORMANCE
    max_concurrent_calls = 50,  #The maximum number of concurrent calls allowed during processing.
    add_questions = [
        'Search for data preparation steps in the code'
    ],  #List of aditional questions to use in STEP 1
    device = "cpu", # Select device "cpu" or "auto"
)
```

### Section specifiying
The content of the resulting readme file can be highly customized by building a json file with the structure:

```python
sections = {
    'section_1': {
        'general_description': 'full text with the description of the section',
        'subsections': {
            'subsection_1_1': 'full text with the description of the subsection_1_1',
            'subsection_1_2': 'full text with the description of the subsection_1_2',
            ...
        }
    },
    'section_2': {
        'general_description': 'full text with the description of the section',
        'subsections': {
            'subsection_2_1': 'full text with the description of the subsection_2_1',
            'subsection_2_2': 'full text with the description of the subsection_2_2',
            ...
        }
    },
    ...
}
```
The proposed structure for a good handled json is shown below:

```python
sections = {
        'Description': {
            'general_description': 'Take into account that this repo is a backend for a chat agent dedicated to answer questions about a mongodb database. Use this context to provide deep details based in repo content',
            'subsections': {
                'Kafka': 'Prepare a content inside #Description section talking about the ios folder-Kafka',
                'MongoDB': 'Prepare a content inside #Description section talking about the ios folder-MongoDB',
                'Azure AppInsigths': 'Prepare a content inside #Description section talking about the ios folder-Azure AppInsigths',
                'Redis': 'We are using Redis to provide chat history and enhance the agent capabilities. Prepare a content inside #Description',
                'LangGraph': 'Prepare a content inside #Description section talking about the LangGraph and its roll in the development',
                'Agent Tools': 'List the tools inside src/tools and provide a deep description for each one providing some examples about the context in which will be useful each set of tools'
            }
        },
        'Requirements': {
            'general_description': 'Provide general overview of required libraries. Remember we use python 3.10 in this project',
            'subsections': {
                'Environment setup': 'Provide details on how to setup a python environment',
                'Requirements installing': 'Provide details about libraries consider in this development using as source the requirements.txt file. Remember that core libraries are pymongo, langgraph, langchain, redis, networkx and fastapi. Provide details on how to use requirements on the setup',
                'Environment variables': f'The list of environment variables are: {coma_separated_environment_variables}.\n Provide brief definition for each variable.'
            }
        },
        'Usage': {
            'general_description': 'The usage of this project is basically enrolled in a chat application.',
            'subsections': {
                'Fake events': 'To locally run we need to provide fake events to simulate cloud behaviour (user_prompt.json and agent_response.json)',
                'How to run': 'Once we have al environment setup we can run the code with ´python src/__main__.py´ command.'
            }
        }
    }
```