#!/usr/bin/env python
"""
Gradio UI for RAG-Anything with Azure OpenAI
Simple interface to upload documents and query them
"""

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

from raganything import RAGAnything
from config import RAGAnythingConfig
from azure_openai_wrappers import create_azure_llm_func, create_azure_embedding_func, create_azure_vision_func

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)

# Set up logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gradio_ui.log')
    ]
)
logger = logging.getLogger(__name__)

# Global RAG instance
rag_instance = None

def initialize_rag():
    """Initialize RAG-Anything with Azure OpenAI"""
    global rag_instance
    
    if rag_instance is None:
        config = RAGAnythingConfig(
            working_dir="./rag_ui_storage",
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
            display_content_stats=True
        )
        
        rag_instance = RAGAnything(
            config=config,
            llm_model_func=create_azure_llm_func(),
            embedding_func=create_azure_embedding_func(),
            vision_model_func=create_azure_vision_func()
        )
    
    return rag_instance

async def process_document_async(file_path):
    """Process a document asynchronously"""
    rag = initialize_rag()
    await rag.process_document_complete(
        file_path=file_path,
        output_dir="./parsed_output_ui"
    )

def get_document_id(file_path):
    """Generate a document ID from a file path"""
    rag = initialize_rag()
    file_path_obj = Path(file_path)
    mtime = file_path_obj.stat().st_mtime
    config_str = json.dumps({"file_path": str(file_path_obj.absolute()), "mtime": mtime})
    return f"doc-{hashlib.md5(config_str.encode()).hexdigest()}"

async def check_document_status(file_path):
    """Check if a document has already been processed"""
    rag = initialize_rag()
    await rag._ensure_lightrag_initialized()
    
    doc_id = get_document_id(file_path)
    
    is_processed = await rag.is_document_fully_processed(doc_id)
    
    return is_processed

async def process_document(file):
    """Process uploaded document"""
    if file is None:
        return "Please upload a document first."
    
    file_path = file.name
    file_name = os.path.basename(file_path)
    
    try:
        # Check if the document is already processed
        is_processed = await check_document_status(file_path)
        
        if is_processed:
            return f"✅ Document '{file_name}' has already been processed and is ready for querying."
        
        # If not processed, run the async processing
        await process_document_async(file_path)
        return f"✅ Successfully processed: {file_name}"
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return f"❌ Error processing document: {str(e)}"

async def query_async(question, timeout_seconds=60):
    """Query the processed documents asynchronously with timeout"""
    import time
    start_time = time.time()
    
    logger.info(f"Starting query: {question[:100]}...")
    rag = initialize_rag()
    
    try:
        # Ensure LightRAG is initialized before querying
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

async def query_documents(question, timeout_seconds=60):
    """Query the processed documents with configurable timeout"""
    if not question:
        return "Please enter a question."
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"New query request: {question}")
        logger.info(f"Timeout set to: {timeout_seconds} seconds")
        logger.info(f"{'='*60}")
        
        # Check if storage exists
        storage_path = "./rag_ui_storage"
        if not os.path.exists(storage_path):
            return "❌ No documents have been processed yet. Please upload and process a document first."
        
        # Check if any documents are indexed
        chunks_file = os.path.join(storage_path, "vdb_chunks.json")
        if not os.path.exists(chunks_file):
            return "❌ No documents in the index. Please process a document first."
        
        # Run async query with timeout
        try:
            result = await query_async(question, timeout_seconds=timeout_seconds)
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

async def process_existing_pdf():
    """Process the existing product.pdf file"""
    pdf_path = "output/product.pdf"
    if os.path.exists(pdf_path):
        try:
            await process_document_async(pdf_path)
            return f"✅ Successfully processed: {pdf_path}"
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return f"❌ Error processing PDF: {str(e)}"
    else:
        return f"❌ File not found: {pdf_path}"

# Create Gradio interface
with gr.Blocks(title="RAG-Anything UI") as demo:
    gr.Markdown("""
    # 🚀 RAG-Anything with Azure OpenAI
    
    Upload documents and ask questions about them using multimodal RAG.
    """)
    
    with gr.Tab("Upload & Process"):
        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="Upload Document",
                    file_types=[".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg"]
                )
                process_btn = gr.Button("Process Document", variant="primary")
                
                # Button to process existing PDF
                gr.Markdown("### Or process existing document:")
                existing_pdf_btn = gr.Button("Process output/product.pdf", variant="secondary")
                
            with gr.Column():
                process_output = gr.Textbox(
                    label="Processing Status",
                    lines=3,
                    interactive=False
                )
        
        process_btn.click(
            fn=process_document,
            inputs=[file_input],
            outputs=[process_output]
        )
        
        existing_pdf_btn.click(
            fn=process_existing_pdf,
            inputs=[],
            outputs=[process_output]
        )
    
    with gr.Tab("Query Documents"):
        with gr.Row():
            with gr.Column():
                question_input = gr.Textbox(
                    label="Ask a Question",
                    placeholder="What is this document about?",
                    lines=2
                )
                
                # Add timeout slider
                with gr.Row():
                    timeout_slider = gr.Slider(
                        minimum=10,
                        maximum=120,
                        value=60,
                        step=10,
                        label="Query Timeout (seconds)",
                        info="Increase for complex queries"
                    )
                
                query_btn = gr.Button("Submit Query", variant="primary")
                
                # Example questions
                gr.Markdown("### Example Questions:")
                example_btns = []
                examples = [
                    "What is the main topic of this document?",
                    "Summarize the key features or findings.",
                    "What data or statistics are mentioned?",
                    "List all the technical specifications.",
                    "What are the main benefits described?"
                ]
                
                for example in examples:
                    btn = gr.Button(example, size="sm", variant="secondary")
                    example_btns.append(btn)
                
            with gr.Column():
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=10,
                    interactive=False
                )
        
        # Connect query button with timeout
        query_btn.click(
            fn=query_documents,
            inputs=[question_input, timeout_slider],
            outputs=[answer_output]
        )
        
        # Connect example buttons with timeout
        for btn, example in zip(example_btns, examples):
            btn.click(
                fn=lambda x=example: x,
                outputs=[question_input]
            ).then(
                fn=query_documents,
                inputs=[question_input, timeout_slider],
                outputs=[answer_output]
            )
    
    with gr.Tab("Info"):
        gr.Markdown("""
        ## 📋 How to Use
        
        1. **Upload Document**: Go to the "Upload & Process" tab and upload your document (PDF, TXT, MD, DOCX, images)
        2. **Process**: Click "Process Document" to parse and index the content
        3. **Query**: Go to "Query Documents" tab and ask questions about your document
        
        ## 🎯 Features
        
        - **Multimodal Processing**: Handles text, tables, images, and equations
        - **Azure OpenAI**: Powered by GPT-4 and text-embedding-3-large
        - **Hybrid Search**: Combines semantic and keyword search
        - **Smart Parsing**: Uses MinerU 2.0 for advanced document understanding
        
        ## ⚙️ Current Configuration
        
        - LLM: Azure OpenAI GPT-4
        - Embeddings: Azure OpenAI text-embedding-3-large
        - Parser: MinerU 2.0
        - Storage: Local (./rag_ui_storage)
        """)

if __name__ == "__main__":
    # Check environment variables
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
    
    print("🚀 Starting RAG-Anything UI...")
    print("📂 Ready to process documents!")
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
