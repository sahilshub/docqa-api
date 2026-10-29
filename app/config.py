from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_api_key: str = ""
    llm_model: str = ""
    
    request_timeout_seconds: int = 30
    model_config = {"env_file": ".env"}

settings = Settings()
