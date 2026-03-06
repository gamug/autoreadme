"""
LLM Utils
"""

import os
import sys

import tiktoken, torch
from huggingface_hub import hf_hub_download
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from autoreadme.types import LLMModelDetails, LLMModels


def get_llm_encoder(model: LLMModels):
    """Get LLM Encoder for token counting"""
    try:
        # Prefer explicit mapping if tiktoken knows the model
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        # Fall back to a chosen base encoding
        encoding =  tiktoken.get_encoding("o200k_base")
    return encoding

def get_gemma_chat_model(model_name: str, streaming=False, model_kwargs=None):
    """Get GEMMA Chat Model"""
    gguf_file = None
    if "gguf_file" in model_kwargs and model_kwargs["gguf_file"] is not None:
        gguf_file = model_kwargs["gguf_file"]
        _ = hf_hub_download(model_name, gguf_file)
    tokenizer = get_tokenizer(model_name, gguf_file)
    if sys.platform != "darwin" and "gptq" not in model_name.lower():
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            gguf_file=gguf_file,
            trust_remote_code=True,
            device_map=model_kwargs["device"],
            quantization_config=bnb_config,
            token=os.environ["HF_TOKEN"],
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            gguf_file=gguf_file,
            trust_remote_code=True,
            device_map=model_kwargs["device"],
            token=os.environ["HF_TOKEN"],
        )

    if (
        "peft_model_path" in model_kwargs
        and model_kwargs["peft_model_path"] is not None
    ):
        PEFT_MODEL = model_kwargs["peft_model_path"]
        model = PeftModel.from_pretrained(model, PEFT_MODEL)

    print(
        f"Memory footprint: {model.get_memory_footprint() / 1024 **3:.2f} GB."
    )

    return HuggingFacePipeline(
        pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=512,
        ),
        model_kwargs=model_kwargs,
    )


def get_llama_chat_model(model_name: str, streaming=False, model_kwargs=None):
    """Get LLAMA2 Chat Model"""
    gguf_file = None
    if "gguf_file" in model_kwargs and model_kwargs["gguf_file"] is not None:
        gguf_file = model_kwargs["gguf_file"]
        _ = hf_hub_download(model_name, gguf_file)
    tokenizer = get_tokenizer(model_name, gguf_file)
    tokenizer.pad_token = tokenizer.eos_token
    if sys.platform != "darwin" and "gptq" not in model_name.lower():
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            # use_exllama=False,
            # exllama_config={"version": 2}
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            gguf_file=gguf_file,
            trust_remote_code=True,
            device_map=model_kwargs["device"],
            quantization_config=bnb_config,
            token=os.environ["HF_TOKEN"],
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            gguf_file=gguf_file,
            trust_remote_code=True,
            device_map=model_kwargs["device"],
            token=os.environ["HF_TOKEN"],
        )

    if "peft_model" in model_kwargs and model_kwargs["peft_model"] is not None:
        PEFT_MODEL = model_kwargs["peft_model"]
        model = PeftModel.from_pretrained(model, PEFT_MODEL)

    print(
        f"Memory footprint: {model.get_memory_footprint() / 1024 **3:.2f} GB."
    )

    return HuggingFacePipeline(
        pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            repetition_penalty=1.2,
            return_full_text=False,
            max_new_tokens=512,
        ),
        model_kwargs=model_kwargs,
    )


def get_ollama_chat_model(model_name: str, streaming=False, model_kwargs=None):
    """Get Ollama Chat Model"""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model_name,
        temperature=model_kwargs["temperature"],
        num_ctx=model_kwargs["max_length"],
        disable_streaming=not streaming,
    )


def get_openai_chat_model() -> AzureChatOpenAI:
    """Get OpenAI Chat Model configured for Azure OpenAI Service with reasoning controls"""
    doc_chat_model = AzureChatOpenAI(
        openai_api_key=os.environ['LLM_API_KEY'],
        azure_endpoint=os.environ['LLM_API_URI'],
        azure_deployment=os.environ['LLM_DEPLOYMENT'],
        api_version=os.environ['LLM_API_VERSION'],
        use_responses_api=True,  # <- important for reasoning controls
        use_previous_response_id=False,  # <- lets the service carry prior reasoning
    )
    return doc_chat_model

def get_tokenizer(model_name: str, gguf_file=None):
    """Get Tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        gguf_file=gguf_file,
        token=os.environ["HF_TOKEN"],
        # use_fast=True
    )
    return tokenizer


models = {
    LLMModels.GPT52: LLMModelDetails(
        name=LLMModels.GPT52,
        input_cost_per_1k_tokens=0.00175,
        output_cost_per_1k_tokens=0.014,
        max_length=200000 ,
        llm=get_openai_chat_model(),
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
    ),
    LLMModels.GPT5: LLMModelDetails(
        name=LLMModels.GPT5,
        input_cost_per_1k_tokens=0.00125,
        output_cost_per_1k_tokens=0.01,
        max_length=400000 ,
        llm=get_openai_chat_model(),
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
    ),
    LLMModels.GPT3: LLMModelDetails(
        name=LLMModels.GPT3,
        input_cost_per_1k_tokens=0.0015,
        output_cost_per_1k_tokens=0.002,
        max_length=3050,
        llm=ChatOpenAI(
            temperature=0.1,
            api_key=os.environ['LLM_API_KEY'],
            model=LLMModels.GPT3,
        ),
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
    ),
    LLMModels.GPT4: LLMModelDetails(
        name=LLMModels.GPT4,
        input_cost_per_1k_tokens=0.03,
        output_cost_per_1k_tokens=0.06,
        max_length=8192,
        llm=ChatOpenAI(
            temperature=0.1,
            api_key=os.environ['LLM_API_KEY'],
            model=LLMModels.GPT4,
        ),
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
    ),
    LLMModels.GPT432k: LLMModelDetails(
        name=LLMModels.GPT432k,
        input_cost_per_1k_tokens=0.06,
        output_cost_per_1k_tokens=0.12,
        max_length=32768,
        llm=ChatOpenAI(
            temperature=0.1,
            api_key=os.environ['LLM_API_KEY'],
            model=LLMModels.GPT4,
        ),
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
    ),
    LLMModels.TINYLLAMA_1p1B_CHAT_GGUF: LLMModelDetails(
        name=LLMModels.TINYLLAMA_1p1B_CHAT_GGUF,
        input_cost_per_1k_tokens=0,
        output_cost_per_1k_tokens=0,
        max_length=2048,
        llm=None,
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
        gguf_file="tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
    ),
    LLMModels.LLAMA2_7B_CHAT_GPTQ: LLMModelDetails(
        name=LLMModels.LLAMA2_7B_CHAT_GPTQ,
        input_cost_per_1k_tokens=0,
        output_cost_per_1k_tokens=0,
        max_length=4096,
        llm=None,
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
    ),
    LLMModels.LLAMA2_13B_CHAT_GPTQ: LLMModelDetails(
        name=LLMModels.LLAMA2_13B_CHAT_GPTQ,
        input_cost_per_1k_tokens=0,
        output_cost_per_1k_tokens=0,
        max_length=4096,
        llm=None,
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
    ),
    LLMModels.CODELLAMA_7B_INSTRUCT_GPTQ: LLMModelDetails(
        name=LLMModels.CODELLAMA_7B_INSTRUCT_GPTQ,
        input_cost_per_1k_tokens=0,
        output_cost_per_1k_tokens=0,
        max_length=8192,
        llm=None,
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
    ),
    LLMModels.CODELLAMA_13B_INSTRUCT_GPTQ: LLMModelDetails(
        name=LLMModels.CODELLAMA_13B_INSTRUCT_GPTQ,
        input_cost_per_1k_tokens=0,
        output_cost_per_1k_tokens=0,
        max_length=8192,
        llm=None,
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
    ),
    LLMModels.LLAMA2_7B_CHAT_HF: LLMModelDetails(
        name=LLMModels.LLAMA2_7B_CHAT_HF,
        input_cost_per_1k_tokens=0,
        output_cost_per_1k_tokens=0,
        max_length=4096,
        llm=None,
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
    ),
    LLMModels.LLAMA2_13B_CHAT_GPTQ: LLMModelDetails(
        name=LLMModels.LLAMA2_13B_CHAT_GPTQ,
        input_cost_per_1k_tokens=0,
        output_cost_per_1k_tokens=0,
        max_length=4096,
        llm=None,
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
    ),
    LLMModels.CODELLAMA_7B_INSTRUCT_HF: LLMModelDetails(
        name=LLMModels.CODELLAMA_7B_INSTRUCT_HF,
        input_cost_per_1k_tokens=0,
        output_cost_per_1k_tokens=0,
        max_length=8192,
        llm=None,
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
    ),
    LLMModels.CODELLAMA_13B_INSTRUCT_HF: LLMModelDetails(
        name=LLMModels.CODELLAMA_13B_INSTRUCT_HF,
        input_cost_per_1k_tokens=0,
        output_cost_per_1k_tokens=0,
        max_length=8192,
        llm=None,
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
    ),
    LLMModels.GOOGLE_GEMMA_2B_INSTRUCT: LLMModelDetails(
        name=LLMModels.GOOGLE_GEMMA_2B_INSTRUCT,
        input_cost_per_1k_tokens=0,
        output_cost_per_1k_tokens=0,
        max_length=8192,
        llm=None,
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
    ),
    LLMModels.GOOGLE_GEMMA_7B_INSTRUCT: LLMModelDetails(
        name=LLMModels.GOOGLE_GEMMA_7B_INSTRUCT,
        input_cost_per_1k_tokens=0,
        output_cost_per_1k_tokens=0,
        max_length=8192,
        llm=None,
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
    ),
    LLMModels.GOOGLE_GEMMA_2B_INSTRUCT_GGUF: LLMModelDetails(
        name=LLMModels.GOOGLE_GEMMA_2B_INSTRUCT_GGUF,
        input_cost_per_1k_tokens=0,
        output_cost_per_1k_tokens=0,
        max_length=8192,
        llm=None,
        input_tokens=0,
        output_tokens=0,
        succeeded=0,
        failed=0,
        total=0,
        gguf_file="gemma-2-2b-it-IQ3_M.gguf",
    ),
}


def print_model_details(models):
    """Print Model Details"""
    output = []
    for model_details in models.values():
        result = {
            "Model": model_details.name,
            "File Count": model_details.total,
            "Succeeded": model_details.succeeded,
            "Failed": model_details.failed,
            "Tokens": model_details.input_tokens + model_details.output_tokens,
            "Cost": (
                (model_details.input_tokens / 1000)
                * model_details.input_cost_per_1k_tokens
                + (model_details.output_tokens / 1000)
                * model_details.output_cost_per_1k_tokens
            ),
        }
        output.append(result)

    totals = {
        "Model": "Total",
        "File Count": sum(item["File Count"] for item in output),
        "Succeeded": sum(item["Succeeded"] for item in output),
        "Failed": sum(item["Failed"] for item in output),
        "Tokens": sum(item["Tokens"] for item in output),
        "Cost": sum(item["Cost"] for item in output),
    }

    all_results = output + [totals]
    for item in all_results:
        print(item)

def get_embeddings(model: str, device: str | None):
    """Get Embeddings"""
    if device == "auto":
        device = None
    if "llama" in model.lower() or "gemma" in model.lower():
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        return AzureOpenAIEmbeddings(
            api_key=os.environ["LLM_API_KEY"],
            azure_endpoint=os.environ["LLM_API_URI"],
            api_version=os.environ.get("LLM_EMBEDDING_API_VERSION", "2024-10-21"),
            model=os.environ.get("LLM_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
        )
