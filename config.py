from dataclasses import dataclass, field
from typing import List, Optional

AVAILABLE_MODELS = ["gpt-4.1-t93a-temp", "gpt-5"]

@dataclass
class RAGAnythingConfig:
    working_dir: str = "./rag_ui_storage"
    parser: str = "mineru"
    parse_method: str = "auto"
    enable_image_processing: bool = True
    enable_table_processing: bool = True
    enable_equation_processing: bool = True
    display_content_stats: bool = True
    llm_model_name: Optional[str] = None
    context_window: int = 2
    context_mode: str = "before"
    max_context_tokens: int = 512
    include_headers: bool = True
    include_captions: bool = True
    context_filter_content_types: List[str] = field(default_factory=lambda: ["text"])
    max_concurrent_files: int = 10
    supported_file_extensions: List[str] = field(default_factory=lambda: [".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg"])
    recursive_folder_processing: bool = True
    parser_output_dir: str = "./parsed_output"
