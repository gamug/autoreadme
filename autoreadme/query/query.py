"""
Utility to Query a Code Repository or Generate README
"""

import json, os
import traceback
from langchain_core.runnables.base import Runnable

from prompt_toolkit.shortcuts import clear

from autoreadme.query.create_chat_chain import make_readme_chain
from autoreadme.types import (
    AutodocReadmeConfig,
    AutodocRepoConfig,
    AutodocUserConfig,
)
from autoreadme.utils.HNSWLib import HNSWLib
from autoreadme.utils.llm_utils import get_embeddings
from autoreadme.utils.file_utils import remove_building_files

chat_history: list[tuple[str, str]] = []

def init_readme_chain(
    repo_config: AutodocRepoConfig, user_config: AutodocUserConfig
):
    data_path = os.path.join(repo_config.output, "docs", "data")
    embeddings = get_embeddings(repo_config.llms[0].name, repo_config.device)
    vector_store = HNSWLib.load(data_path, embeddings)
    chain = make_readme_chain(
        repo_config.name,
        repo_config.repository_url,
        repo_config.content_type,
        repo_config.chat_prompt,
        vector_store,
        user_config.llms,
        repo_config.peft_model_path,
        device=repo_config.device,
        on_token_stream=user_config.streaming,
    )

    return chain

def use_chain(chain: Runnable, query: str, file) -> None:
    try:
        response = chain.invoke({"input": query})
        # print("\n\nMarkdown:\n")
        # print(markdown(response["answer"]))
        if isinstance(response["answer"], dict):
            response = list(response["answer"].values())[0]
        elif isinstance(response["answer"], str):
            try:
                response = list(json.loads(response["answer"]).values())[0]
            except:
                response = response["answer"]
        # response = response.replace('<code>', '\n```').replace('</code>', '\n```')
        # response = response.replace('<pre>', '').replace('</pre>', '')
        # response = response.replace('\n```', '\n\n```')
        file.write(response)
    except RuntimeError as error:
        print(f"Something went wrong: {error}")
        traceback.print_exc()

def generate_readme(
    repo_config: AutodocRepoConfig,
    user_config: AutodocUserConfig,
    readme_config: AutodocReadmeConfig,
    clean_repo: bool=True
):
    """
    Generates a README file based on repository and user configurations.

    Initializes a README generation chain, clears the terminal, and prepares
    the output file. Iterates over the specified headings in the README
    configuration, generates content for each section by invoking the chain,
    and writes the content in Markdown format to the README file. Handles any
    RuntimeError that occurs during the process.

    Args:
        repo_config: An AutodocRepoConfig instance containing configuration
            settings for the repository.
        user_config: An AutodocUserConfig instance containing user-specific
            configuration settings.
        readme_config: An AutodocReadmeConfig instance containing
            configuration settings for README generation.

    """
    chain = init_readme_chain(repo_config, user_config)

    clear()

    print("Generating README...")
    data_path = os.path.join(repo_config.output, "docs", "data")
    readme_path = os.path.join(
        data_path, f"README_{repo_config.llms[0].name}.md"
    )
    with open(readme_path, "w", encoding="utf-8") as file:
        file.write(f"# {repo_config.name}\n")

    with open(readme_path, "a", encoding="utf-8") as file:
        sections = readme_config.sections
        for section in sections:
            print(f'\tprocessing {section.name} section...')
            file.write(f"\n\n## {section.name}\n")
            prompt = f'Provide a general description detailed for the section {section.name}, knowing that {section.prompt}. Provide the description in markdown style without the section-subsection header.'
            use_chain(chain, prompt, file)
            for subsection, description in section.subsections.items():
                print(f'\t\tprocessing {subsection} subsection...')
                file.write(f"\n\n### {subsection}\n")
                prompt = f'In the context of the section {section.name}, expand the explanation for a subsection called {subsection} that must fit the instructions {description}. Provide the description in markdown style without the section-subsection header.'
                use_chain(chain, prompt, file)
            
    if clean_repo:
        print('cleaning repository from building files')
        remove_building_files('.', repo_config)
    
    print(f"README generated at {readme_path}")