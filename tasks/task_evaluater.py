from pathlib import Path
import importlib

class TaskEvaluater:
    def __init__(self, task_name:str, method:str) -> None:
        if task_name in ['Structured_Amazon-Google', 'Structured_Walmart-Amazon', 'Textual_Abt-Buy', 'AAS_ECLASS_new'] and 'selection' in method:
            task_name = 'SelectiveEntityMatching'
        pck = importlib.import_module(f"tasks.{task_name}.{method}")
        evaluater = getattr(pck, 'Evaluater')
        self.task_name = task_name
        self.method = method
        self.evaluater = evaluater()



def main():
    ''
if __name__ == "__main__":
    main()