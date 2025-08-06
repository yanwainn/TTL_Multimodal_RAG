# RAG-Anything

This project is a multimodal RAG (Retrieval-Augmented Generation) application that can process and query various document types, including PDFs, text files, and images. It uses Azure OpenAI for language and vision models and Gradio for the user interface.

## Features

- **Multimodal RAG**: Process and query text, images, and tables from your documents.
- **Azure OpenAI Integration**: Leverages the power of Azure OpenAI for language and vision tasks.
- **Gradio UI**: An easy-to-use web interface for uploading documents and asking questions.
- **Batch Processing**: Process multiple files at once.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yanwainn/TTL_RAG_Everything.git
   cd RAG_Every
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Set up your environment variables:**
   - Create a `.env` file by copying the `.env.sample_azure` file:
     ```bash
     cp .env.sample_azure .env
     ```
   - Edit the `.env` file and add your Azure OpenAI credentials.

4. **Run the Gradio UI:**
   ```bash
   ./run_gradio.sh
   ```

   The UI will be available at `http://localhost:7860`.

## Usage

1. **Upload a document**: Use the "Upload & Process" tab to upload a PDF, text file, or image.
2. **Process the document**: Click the "Process Document" button to parse and index the content.
3. **Query the document**: Go to the "Query Documents" tab and ask questions about the document.
