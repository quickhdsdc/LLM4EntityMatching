from pathlib import Path
import importlib

class TaskDataPreprocessor:
    def __init__(self, task_name:str, method:str) -> None:
        if task_name in ['Structured_Amazon-Google', 'Structured_Walmart-Amazon', 'Textual_Abt-Buy', 'AAS_ECLASS_new'] and 'selection' in method:
            task_name = 'SelectiveEntityMatching'
        pck = importlib.import_module(f"tasks.{task_name}.{method}")
        print(f'import tasks.{task_name}.{method}')
        data_preprocessor = getattr(pck, 'DataPreprocessor')
        if 'Structured' in task_name or 'Textual' in task_name or 'Dirty' in task_name:
            data_preprocessor_pairwise = getattr(pck, 'DataPreprocessor_pairwise')
            self.data_preprocessor_pairwise = data_preprocessor_pairwise()
        self.task_name = task_name
        self.method = method
        self.data_preprocessor = data_preprocessor()
        



def main():
    ''
if __name__ == "__main__":
    main()