import gradio as gr
from prompt import PROMPTS

def create_training_ui(
    get_extracted_files_func,
    train_documents_func,
    update_prompts_func,
    available_models
):
    """Creates the Gradio UI for the training process."""
    with gr.Tab("Train on Extracted Documents"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 1. Select Model")
                model_selector = gr.Dropdown(
                    label="Select Model",
                    choices=available_models,
                    value=available_models[0] if available_models else None,
                    interactive=True
                )

                gr.Markdown("### 2. Select Extracted Files for Training")
                extracted_files_checkboxes = gr.CheckboxGroup(
                    label="Available Extracted Files",
                    info="Select the files you want to train on.",
                    value=[]
                )
                refresh_btn = gr.Button("Refresh File List")
                
                gr.Markdown("---")
                gr.Markdown("### 3. Configure & Start Training")
                train_btn = gr.Button("Run Training on Selected Files", variant="primary")
                
            with gr.Column(scale=3):
                training_status_output = gr.Textbox(
                    label="Training Status Summary",
                    lines=3,
                    interactive=False,
                    placeholder="A summary of the training process will appear here..."
                )
                training_log_output = gr.Textbox(
                    label="Real-Time Training Log",
                    lines=15,
                    interactive=False,
                    placeholder="Detailed logs will stream here during training..."
                )

    with gr.Tab("Prompt Settings"):
        gr.Markdown("## ⚙️ Customize Analysis Prompts")
        gr.Markdown("Modify the prompts used for analyzing different types of content. Changes will be applied when you next run training.")
        
        with gr.Accordion("Image Analysis Prompt", open=False):
            vision_prompt_input = gr.Textbox(
                label="Vision Prompt",
                value=PROMPTS["vision_prompt"],
                lines=15,
                interactive=True
            )
        
        with gr.Accordion("Table Analysis Prompt", open=False):
            table_prompt_input = gr.Textbox(
                label="Table Prompt",
                value=PROMPTS["table_prompt"],
                lines=15,
                interactive=True
            )
            
        with gr.Accordion("Equation Analysis Prompt", open=False):
            equation_prompt_input = gr.Textbox(
                label="Equation Prompt",
                value=PROMPTS["equation_prompt"],
                lines=15,
                interactive=True
            )
            
        with gr.Accordion("Generic Content Prompt", open=False):
            generic_prompt_input = gr.Textbox(
                label="Generic Prompt",
                value=PROMPTS["generic_prompt"],
                lines=15,
                interactive=True
            )
        
        prompt_inputs = [vision_prompt_input, table_prompt_input, equation_prompt_input, generic_prompt_input]
        
        # Update prompts on change
        for prompt_input in prompt_inputs:
            prompt_input.change(
                fn=update_prompts_func,
                inputs=prompt_inputs
            )

    # Connect buttons to functions
    refresh_btn.click(
        fn=lambda model: gr.update(choices=get_extracted_files_func(model)),
        inputs=[model_selector],
        outputs=[extracted_files_checkboxes]
    )
    
    model_selector.change(
        fn=lambda model: gr.update(choices=get_extracted_files_func(model), value=[]),
        inputs=[model_selector],
        outputs=[extracted_files_checkboxes]
    )

    train_btn.click(
        fn=train_documents_func,
        inputs=[model_selector, extracted_files_checkboxes] + prompt_inputs,
        outputs=[training_status_output, training_log_output]
    )
    
    return extracted_files_checkboxes
