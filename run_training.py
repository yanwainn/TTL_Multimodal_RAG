import os
import asyncio
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv
import logging
import json

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from raganything import RAGAnything
from config import RAGAnythingConfig, AVAILABLE_MODELS
from azure_openai_wrappers import create_azure_llm_func, create_azure_embedding_func, create_azure_vision_func, create_gpt5_llm_func
from training_ui import create_training_ui

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

class GradioLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_stream = ""

    def emit(self, record):
        log_entry = self.format(record)
        self.log_stream += log_entry + "\n"

    def get_logs(self):
        return self.log_stream

    def clear_logs(self):
        self.log_stream = ""

gradio_log_handler = GradioLogHandler()
logging.getLogger().addHandler(gradio_log_handler)

# Global RAG instance
rag_instances = {}

def initialize_rag(model_name, custom_prompts=None):
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
            vision_model_func=create_azure_vision_func(deployment_name=model_name),
            prompts=custom_prompts
        )
    
    elif custom_prompts:
        rag_instances[model_name].update_prompts(custom_prompts)
        
    return rag_instances[model_name]

def get_extracted_files(model_name):
    """Scan the parsed_output_ui directory for extracted content lists"""
    output_dir = Path(f"./parsed_output_ui/{model_name}")
    if not output_dir.exists():
        return []
    
    content_files = []
    for file_path in output_dir.rglob("*_content_list.json"):
        content_files.append(str(file_path))
        
    return content_files

async def train_documents(model_name, selected_files, vision_prompt, table_prompt, equation_prompt, generic_prompt):
    """Train on the selected extracted documents and stream logs"""
    if not selected_files:
        yield "Please select one or more extracted files to train on.", ""
        return

    gradio_log_handler.clear_logs()
    
    custom_prompts = {
        "vision_prompt": vision_prompt,
        "table_prompt": table_prompt,
        "equation_prompt": equation_prompt,
        "generic_prompt": generic_prompt,
    }
    
    rag = initialize_rag(model_name, custom_prompts=custom_prompts)
    
    logger.info("="*50)
    logger.info("Starting new training session")
    logger.info("="*50)
    
    # Log configuration
    logger.info("Configuration Parameters:")
    config_info = rag.get_config_info()
    logger.info(json.dumps(config_info, indent=2))
    
    total_files = len(selected_files)
    processed_count = 0
    
    for i, file_path in enumerate(selected_files):
        file_name = Path(file_path).stem.replace('_content_list', '')
        logger.info(f"--- Processing file {i+1}/{total_files}: {file_name} ---")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content_list = json.load(f)
            
            await rag.insert_content_list(
                content_list=content_list,
                file_path=file_name
            )
            logger.info(f"✅ Successfully trained on: {file_name}")
            processed_count += 1
        except Exception as e:
            logger.error(f"❌ Error training on: {file_name} - {e}", exc_info=True)
        
        yield f"Processed {i+1}/{total_files} files...", gradio_log_handler.get_logs()

    summary = f"Training complete. Successfully processed {processed_count}/{total_files} files."
    logger.info(summary)
    logger.info("="*50)
    
    yield summary, gradio_log_handler.get_logs()

def update_prompts(model_name, vision_prompt, table_prompt, equation_prompt, generic_prompt):
    """Update the prompts in the RAG instance"""
    custom_prompts = {
        "vision_prompt": vision_prompt,
        "table_prompt": table_prompt,
        "equation_prompt": equation_prompt,
        "generic_prompt": generic_prompt,
    }
    initialize_rag(model_name, custom_prompts=custom_prompts)
    logger.info("Prompts updated successfully.")

with gr.Blocks(title="Training UI") as demo:
    gr.Markdown("# 🚀 Train on Extracted Documents")
    gr.Markdown("Select the extracted document content you want to process with LightRAG for entity extraction and indexing.")
    
    checkboxes = create_training_ui(
        get_extracted_files_func=get_extracted_files,
        train_documents_func=train_documents,
        update_prompts_func=update_prompts,
        available_models=AVAILABLE_MODELS
    )
    
    demo.load(
        fn=lambda: gr.update(choices=get_extracted_files(AVAILABLE_MODELS[0])),
        outputs=[checkboxes]
    )

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
    
    print("🚀 Starting Training UI...")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False
    )
