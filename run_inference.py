import os
import asyncio
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv
import logging

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from raganything import RAGAnything
from config import RAGAnythingConfig, AVAILABLE_MODELS
from azure_openai_wrappers import create_azure_llm_func, create_azure_embedding_func, create_azure_vision_func, create_gpt5_llm_func
from inference_ui import create_inference_ui

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('inference.log')
    ]
)
logger = logging.getLogger(__name__)

# Global RAG instance
rag_instances = {}

def initialize_rag(model_name):
    """Initialize RAG-Anything with Azure OpenAI"""
    global rag_instances
    
    if model_name not in rag_instances:
        config = RAGAnythingConfig(
            working_dir=f"./rag_ui_storage/{model_name}",
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
            display_content_stats=True,
            llm_model_name=model_name
        )
        
        if model_name == "gpt-5":
            llm_func = create_gpt5_llm_func(deployment_name=model_name)
        else:
            llm_func = create_azure_llm_func(deployment_name=model_name)

        rag_instances[model_name] = RAGAnything(
            config=config,
            llm_model_func=llm_func,
            embedding_func=create_azure_embedding_func(),
            vision_model_func=create_azure_vision_func(deployment_name=model_name)
        )
        
    return rag_instances[model_name]

async def query_async(model_name, question, timeout_seconds=60):
    """Query the processed documents asynchronously with timeout"""
    import time
    start_time = time.time()
    
    logger.info(f"Starting query: {question[:100]}...")
    rag = initialize_rag(model_name)
    
    try:
        logger.info("Initializing LightRAG...")
        await asyncio.wait_for(
            rag._ensure_lightrag_initialized(),
            timeout=10
        )
        logger.info("LightRAG initialized successfully")
        
        mode = "hybrid"
        
        logger.info(f"Executing query in {mode} mode...")
        result = await asyncio.wait_for(
            rag.aquery(
                question,
                mode=mode,
                enable_rerank=False
            ),
            timeout=timeout_seconds - (time.time() - start_time)
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Query completed in {elapsed:.2f} seconds. Result length: {len(result)}")
        return result
        
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.error(f"Query timed out after {elapsed:.2f} seconds")
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Query failed after {elapsed:.2f} seconds: {str(e)}")
        raise

async def query_documents(model_name, question, timeout_seconds=60):
    """Query the processed documents with configurable timeout"""
    if not question:
        return "Please enter a question."
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"New query request: {question}")
        logger.info(f"Timeout set to: {timeout_seconds} seconds")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'='*60}")
        
        storage_path = f"./rag_ui_storage/{model_name}"
        if not os.path.exists(storage_path):
            return f"❌ No documents have been processed for model '{model_name}' yet. Please upload and process a document first."
        
        chunks_file = os.path.join(storage_path, "vdb_chunks.json")
        if not os.path.exists(chunks_file):
            return f"❌ No documents in the index for model '{model_name}'. Please process a document first."
        
        try:
            result = await query_async(model_name, question, timeout_seconds=timeout_seconds)
            logger.info(f"Returning result to UI: {result[:200]}...")
            return result
        except asyncio.TimeoutError:
            return f"⏱️ Query timed out after {timeout_seconds} seconds. This might be due to:\n" \
                   f"1. Complex query requiring more processing\n" \
                   f"2. Network latency to Azure OpenAI\n" \
                   f"3. Large document corpus\n\n" \
                   f"Try a simpler query or increase timeout."
        
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"❌ Error: {str(e)}\n\nPlease check the logs for more details."

with gr.Blocks(title="Inference UI") as demo:
    gr.Markdown("# 🚀 Query and Inference")
    create_inference_ui(query_documents, AVAILABLE_MODELS)

if __name__ == "__main__":
    required_vars = [
        "LLM_BINDING_API_KEY",
        "LLM_BINDING_HOST",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these in your .env file.")
        exit(1)
    
    print("🚀 Starting Inference UI...")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )
