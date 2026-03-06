import os
from dotenv import load_dotenv
load_dotenv()

from autoreadme.types import (
    AutodocReadmeConfig,
    AutodocRepoConfig,
    AutodocUserConfig
)

from autoreadme.utils.file_utils import get_environment_variables
from autoreadme.types import LLMModels, Priority
from autoreadme.query import query
from autoreadme.index import index

llm = LLMModels.GPT5

repo_config = AutodocRepoConfig (
    name = "Canvas Objects", # Replace <REPOSITORY_NAME>
    root = "C:/Users/MP375VC/OneDrive - EY/Desktop/Canvas/canvas-objects", # Replace <REPOSITORY_ROOT_DIR_PATH>
    repository_url = "https://github.com/ey-org/canvas-objects/tree/dev", # Replace <REPOSITORY_URL>
    output = "scripts/readme_generation/readme_generated", # Replace <OUTPUT_DIR_PATH>
    llms = [llm],
    ignore = [
        ".*",
        "*database_schema.json",
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
        "database_schemas*",
        ".pytest_cache*",
        ".vscode*",
        "documentation*",
        "fake_events*",
        "tool_output*",
        "*stability_test*",
        "Adversarial dataset.xlsx",
        "*readme_generation*",
        "*canvas_objects.log"
    ],
    file_prompt = """Write a detailed technical explanation of
            what this code does. Focus on the high-level
            purpose of the code and how it may be used in the
            larger project.\nInclude code examples where
            appropriate. Keep you response between 100 and 300
            words. DO NOT RETURN MORE THAN 300 WORDS.
            Output should be in markdown format.
            Do not just list the methods and classes in this file.""",
    peft_model_path=None,
    folder_prompt = """Write a technical explanation of what the
            code in this file does and how it might fit into the
            larger project or work with other parts of the project.
            Give examples of how this code might be used. Include code
            examples where appropriate. Be concise. Include any
            information that may be relevant to a developer who is
            curious about this code. Keep you response under
            400 words. Output should be in markdown format.
            Do not just list the files and folders in this folder.""",
    chat_prompt = None,
    content_type = "python code",
    target_audience = "smart developer",
    link_hosted = True,
    priority = Priority.PERFORMANCE,
    max_concurrent_calls = 50,
    add_questions = [
        'Prepare a content inside #Description section talking about the ios folder-Kafka',
        'Prepare a content inside #Description section talking about the ios folder-MongoDB',
        'Prepare a content inside #Description section talking about the ios folder-Azure AppInsigths',
        'Prepare a content inside #Description section talking about the ios folder-Redis',
        'Prepare a content inside #Description section talking about the ios folder-OpenAI',
        'Under the #Instalation section describe exaustively the cloning, python setup, requirements description, and environmental variables needs'
    ],
    device = "cpu", # Select device "cpu" or "auto"
)

user_config, environments = AutodocUserConfig(llms=[llm]), get_environment_variables()


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
                'Environment variables': f'The list of environment variables are: {environments}.\n Provide brief definition for each variable.'
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

readme_config = AutodocReadmeConfig(sections=sections)

# index.index(repo_config)
query.generate_readme(repo_config, user_config, readme_config, clean_repo=False)