import gradio as gr

def create_inference_ui(query_documents_func, available_models):
    with gr.Tab("Query Documents"):
        with gr.Row():
            with gr.Column():
                model_selector = gr.Dropdown(
                    label="Select Model",
                    choices=available_models,
                    value=available_models[0] if available_models else None,
                    interactive=True
                )
                question_input = gr.Textbox(
                    label="Ask a Question",
                    placeholder="What is this document about?",
                    lines=2
                )
                
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
        
        query_btn.click(
            fn=query_documents_func,
            inputs=[model_selector, question_input, timeout_slider],
            outputs=[answer_output]
        )
        
        for btn, example in zip(example_btns, examples):
            btn.click(
                fn=lambda x=example: x,
                outputs=[question_input]
            ).then(
                fn=query_documents_func,
                inputs=[model_selector, question_input, timeout_slider],
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
