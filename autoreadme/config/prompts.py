"""
Prompts
"""

from typing import List

from langchain_core.prompts import PromptTemplate
from autoreadme.types import FileSummary, FolderSummary


def create_code_file_summary(
    file_path: str,
    project_name: str,
    file_contents: str,
    content_type: str,
    file_prompt: str,
) -> str:
    """Create Code File Summary"""
    return f"""
    You are acting as a {content_type} documentation expert
    for a project called {project_name}.
    Below is the {content_type} from a file located at `{file_path}`.
    {file_prompt}
    Do not say "this file is a part of the {project_name} project".

    {content_type}:
    {file_contents}

    Response:

    """


def create_code_questions(
    file_path: str,
    project_name: str,
    file_contents: str,
    content_type: str,
    target_audience: str,
) -> str:
    """Create Code Questions"""
    return f"""
    You are acting as a {content_type} documentation expert
    for a project called {project_name}.
    Below is the {content_type} from a file located at `{file_path}`.
    What are 3 questions that a {target_audience} might have about
    this {content_type}?
    Answer each question in 1-2 sentences. Output should be in markdown format.

    {content_type}:
    {file_contents}

    Questions and Answers:

    """


def folder_summary_prompt(
    folder_path: str,
    project_name: str,
    files: List[FileSummary],
    folders: List[FolderSummary],
    content_type: str,
    folder_prompt: str,
) -> str:
    """Folder Summary Prompt"""
    files_summary = "\n".join(
        [
            f"""
        Name: {name}
        Summary: {summary}

      """
            for file in files for name, summary in file.items()
        ]
    )

    folders_summary = "\n".join(
        [
            f"""
        Name: {name}
        Summary: {summary}

      """
            for folder in folders for name, summary in folder.items()
        ]
    )

    return f"""
    You are acting as a {content_type} documentation expert
    for a project called {project_name}.
    You are currently documenting the folder located at `{folder_path}`.

    Below is a list of the files in this folder and a summary
    of the contents of each file:
    {files_summary}

    And here is a list of the subfolders in this folder and a
    summary of the contents of each subfolder:
    {folders_summary}

    {folder_prompt}
    Do not say "this file is a part of the {project_name} project".
    Do not just list the files and folders.

    Response:
    """

def create_readme_prompt(
    project_name, repository_url, content_type, chat_prompt
):
    """Make Readme Prompt"""
    additional_instructions = (
        "\nHere are some additional instructions for "
        + f"generating readme content about {content_type}:\n"
        + f"{chat_prompt}"
        if chat_prompt
        else ""
    )
    template = f"""You are an AI assistant for a software project called {project_name}.
    You are trained on all the {content_type} that makes up this project.
    The {content_type} for the project is located at {repository_url}.
    You are given a repository which might contain several modules and each module will contain a set of files.
    Look at the source code in the repository and generate content for a README.md section under the heading specified below.

    Assumptions:
    - The reader is a smart programmer but not deeply familiar with {project_name}.
    - The reader does not know the project structure, folders/files, or functions.

    INSTRUCTIONS:
    - Do not use <code> nor </pre> tags, use fenced code blocks with ``` instead with code type specified (cmd, bas, java, python, etc).
    - Each time you use ```, do it id a new line and close it in a new line, to avoid formatting issues in the final readme.
    - Only use CONTEXT to build the section content. If CONTEXT is empty output exactly: No content found for this section
    - Replace any mention to repository files with <a></a> html tag using the CONTEXT as source for links, do not invent links.
    - Do not include information that is not directly relevant to the repository.
    - Answer in Markdown without introductory text nor meta commentary.
    - If you need to accentuate use ** for bold and * for italics, do not use html title tags.
    - If you need to introduce bullets use html tags <ul><li>...</li></ul> instead of markdown bullets, to avoid formatting issues in the final readme.
    - Preserve variables and code names, don't change them. Do not replace dounder with <em> markdown or other formatting, keep them as they are in the codebase.
    {additional_instructions}

    QUESTION:
    {{input}}

    CONTEXT:
    {{context}}
    """

    # Return a template object instead of string
    # if you have a class handling it
    return PromptTemplate(
        template=template, input_variables=["input", "context"]
    )
