import os
import openai
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()
openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # Your Azure OpenAI resource's endpoint value.
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
# MODEL_ID = "gpt-4-1106-preview"
# MODEL_ID = "o1-preview-2024-09-12"


class ModelLoader:
    def __init__(self) -> None:
        print('Loading the model...')
    def load_model_from_path_name_version(self):
        model = openai.AzureOpenAI(api_version="2023-05-15")
        return model