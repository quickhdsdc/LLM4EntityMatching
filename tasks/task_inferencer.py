from pathlib import Path
import importlib

class TaskInferencer:
    def __init__(self, task_name:str, method:str) -> None:
        '''
        '''
        pck = importlib.import_module(f"tasks.{task_name}.{method}")
        inferencer = getattr(pck, 'Inferencer')
        self.task_name = task_name
        self.method = method
        self.inferencer = inferencer()



def main():
    ''
if __name__ == "__main__":
    main()