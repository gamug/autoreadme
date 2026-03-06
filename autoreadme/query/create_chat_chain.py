"""
Creates Chains for QA Chat or Readme Generation
"""

from typing import List

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import Runnable

from autoreadme.types import LLMModels
from autoreadme.config.prompts import create_readme_prompt
from autoreadme.utils.HNSWLib import HNSWLib
from autoreadme.utils.llm_utils import (
    get_gemma_chat_model,
    get_llama_chat_model,
    get_openai_chat_model,
    models,
)

# Define the prompt template for condensing the follow-up question
condense_readme_prompt = PromptTemplate.from_template(
    template="Given the following question, rephrase the question "
    + "to be a standalone question.\n\n"
    + "Input: {input}\nStandalone question:"
)

def make_readme_chain(
    project_name: str,
    repository_url: str,
    content_type: str,
    chat_prompt: str,
    vector_store: HNSWLib,
    llms: List[LLMModels],
    peft_model: str | None = None,
    device: str = "cpu",
    on_token_stream: bool = False,
) -> Runnable:
    """
    Creates a README generation chain for the specified project

    Initializes and configures the README generation chain using the provided
    repository, user, and README configurations. Selects the appropriate
    language model (LLM), sets up the document processing chain with the
    specified prompts, and integrates with the vector store to generate
    comprehensive README sections based on project data. The chain facilitates
    automated generation of README files tailored to the project's
    specifications.

    Args:
        project_name: The name of the project for which the README is
            being generated.
        repository_url: The URL of the repository containing the project.
        content_type: The type of content to be included in the README
            (e.g., 'overview', 'installation').
        chat_prompt: The prompt template used for generating README content.
        vector_store: An instance of HNSWLib representing the vector store
            containing document embeddings.
        llms: A list of LLMModels to select from for generating README content.
        peft_model: An optional parameter specifying a PEFT
            (Parameter-Efficient Fine-Tuning) model for enhanced performance.
        device: The device to use for model inference (default is 'cpu').
        on_token_stream: Optional callback for handling token streams during
            model inference.

    Returns:
        A retrieval chain configured for README generation, combining the
            retriever and document processing chain.

    """
    llm = llms[1] if len(llms) > 1 else llms[0]
    llm_name = llm.name
    doc_chat_model = None
    print(f"LLM:  {llm_name.lower()}")
    model_kwargs = {
        "temperature": 0.2,
        "peft_model": peft_model,
        "device": device,
    }
    if "llama" in llm_name.lower():
        if "gguf" in llm_name.lower():
            model_kwargs["gguf_file"] = models[llm].gguf_file
        doc_chat_model = get_llama_chat_model(
            llm_name,
            streaming=bool(on_token_stream),
            model_kwargs=model_kwargs,
        )
    elif "gemma" in llm_name.lower():
        if "gguf" in llm_name.lower():
            model_kwargs["gguf_file"] = models[llm].gguf_file
        doc_chat_model = get_gemma_chat_model(
            llm_name,
            streaming=bool(on_token_stream),
            model_kwargs=model_kwargs,
        )
    else:
        doc_chat_model = get_openai_chat_model()

    readme_prompt = create_readme_prompt(
        project_name,
        repository_url,
        content_type,
        chat_prompt,
    )
    doc_chain = create_stuff_documents_chain(
        llm=doc_chat_model, prompt=readme_prompt
    )

    return create_retrieval_chain(
        retriever=vector_store.as_retriever(), combine_docs_chain=doc_chain
    )