import os
import asyncio
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv
import logging
import hashlib
import json

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from parser import MineruParser
from extraction_ui import create_extraction_ui
from config import AVAILABLE_MODELS

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('extraction.log')
    ]
)
logger = logging.getLogger(__name__)

async def parse_document_async(file_path, model_name):
    """Parse a document asynchronously using MineruParser"""
    parser = MineruParser(deployment_name=model_name)
    output_dir = f"./parsed_output_ui/{model_name}"
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Starting parsing for: {file_path} using model: {model_name}")
    try:
        content_list = await asyncio.to_thread(
            parser.parse_document,
            file_path=file_path,
            output_dir=output_dir,
            method="auto"
        )
        logger.info(f"Successfully parsed {file_path}. Found {len(content_list)} content blocks.")
        return f"✅ Successfully extracted content from: {os.path.basename(file_path)}"
    except Exception as e:
        logger.error(f"Error parsing document {file_path}: {e}", exc_info=True)
        return f"❌ Error extracting content from: {os.path.basename(file_path)} - {e}"

async def process_documents(files, model_name):
    """Process uploaded documents by parsing them"""
    if not files:
        return "Please upload one or more documents first."
    
    tasks = [parse_document_async(file.name, model_name) for file in files]
    results = await asyncio.gather(*tasks)
    return f"Extraction process started for {len(files)} files using model: {model_name}\n\n" + "\n".join(results)

async def process_existing_pdf(model_name):
    """Process the existing product.pdf file by parsing it"""
    pdf_path = "output/product.pdf"
    if os.path.exists(pdf_path):
        return await parse_document_async(pdf_path, model_name)
    else:
        return f"❌ File not found: {pdf_path}"

with gr.Blocks(title="Extraction UI") as demo:
    gr.Markdown("# 🚀 Document Content Extraction (Mineru)")
    gr.Markdown("Upload documents to extract their content into a structured format. The output will be saved in the `parsed_output_ui` directory and can be used for training.")
    create_extraction_ui(process_documents, process_existing_pdf, AVAILABLE_MODELS)

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
    
    print("🚀 Starting Extraction UI...")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
