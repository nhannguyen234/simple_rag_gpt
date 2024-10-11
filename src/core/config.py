import os

class ConfigSettings:
    OPENAI_API_KEY: str = str(os.getenv('OPENAI_API_KEY'))
    CHUNK_SIZE: int = 512
    TOP_K: int = 3

configs = ConfigSettings()