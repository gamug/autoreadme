import json, os
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



#basic input parameters
llm = LLMModels.GPT5
module = "IC"


#Loading repo config from json and updating with necessary fields
with open(os.path.join('json_inputs', f'{module}-repo_config.json'), 'r') as f:
    repo_config_dict = json.load(f)
repo_config_dict['llms'] = [llm]
repo_config_dict['output'] = module
repo_config_dict["file_prompt"] = """Write a detailed technical explanation of
                                    what this code does. Focus on the high-level
                                    purpose of the code and how it may be used in the
                                    larger project.\nInclude code examples where
                                    appropriate. Keep you response between 100 and 300
                                    words. DO NOT RETURN MORE THAN 300 WORDS.
                                    Output should be in markdown format.
                                    Do not just list the methods and classes in this file."""
repo_config_dict["folder_prompt"] = """Write a technical explanation of what the
                                        code in this file does and how it might fit into the
                                        larger project or work with other parts of the project.
                                        Give examples of how this code might be used. Include code
                                        examples where appropriate. Be concise. Include any
                                        information that may be relevant to a developer who is
                                        curious about this code. Keep you response under
                                        400 words. Output should be in markdown format.
                                        Do not just list the files and folders in this folder."""
repo_config_dict["priority"] = Priority.PERFORMANCE
repo_config = AutodocRepoConfig(**repo_config_dict)

#Defining user config and environment variables
user_config, environments = AutodocUserConfig(llms=[llm]), get_environment_variables()

#Defining readme content
with open(os.path.join('json_inputs', f'{module}-sections.json'), 'r') as f:
    sections = json.load(f)
readme_config = AutodocReadmeConfig(sections=sections)

#running the code to generate readme
index.index(repo_config)
query.generate_readme(repo_config, user_config, readme_config, clean_repo=False)