from pathlib import Path
import importlib

class TaskModelLoader:
    def __init__(self, task_name:str, method:str) -> None:
        if task_name in ['Structured_Amazon-Google', 'Structured_Walmart-Amazon', 'Textual_Abt-Buy', 'AAS_ECLASS_new'] and 'selection' in method:
            task_name = 'SelectiveEntityMatching'
        pck = importlib.import_module(f"tasks.{task_name}.{method}")
        model_loader = getattr(pck, 'ModelLoader')
        self.task_name = task_name
        self.method = method
        self.model_loader = model_loader()
    