import gradio as gr

def create_extraction_ui(process_documents_func, process_existing_pdf_func, available_models):
    """Creates the Gradio UI for document extraction."""
    with gr.Tab("Upload & Extract"):
        with gr.Row():
            with gr.Column(scale=2):
                model_selector = gr.Dropdown(
                    label="Select Model",
                    choices=available_models,
                    value=available_models[0] if available_models else None,
                    interactive=True
                )
                file_input = gr.File(
                    label="Upload Documents",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg"]
                )
                process_btn = gr.Button("Extract Document Content", variant="primary")
                
                gr.Markdown("---")
                gr.Markdown("### Or process an existing document:")
                existing_pdf_btn = gr.Button("Process output/product.pdf", variant="secondary")
                
            with gr.Column(scale=3):
                process_output = gr.Textbox(
                    label="Extraction Status",
                    lines=10,
                    interactive=False,
                    placeholder="Extraction status will appear here..."
                )
    
    process_btn.click(
        fn=process_documents_func,
        inputs=[file_input, model_selector],
        outputs=[process_output]
    )
    existing_pdf_btn.click(
        fn=process_existing_pdf_func,
        inputs=[model_selector],
        outputs=[process_output]
    )
