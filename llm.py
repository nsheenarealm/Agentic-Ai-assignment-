"""LLM factory. Supports Gemini (default), Groq, or a local Ollama model."""
import os


class LLMConfigError(Exception):
    pass


def get_llm(provider=None, model=None, api_key=None, temperature=0.3):
    provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMConfigError("No Google API key found. Add GOOGLE_API_KEY to .env or paste it in the sidebar.")
        llm = ChatGoogleGenerativeAI(
            model=model or os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
            temperature=temperature,
            google_api_key=key,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise LLMConfigError("No Groq API key found. Add GROQ_API_KEY to .env or paste it in the sidebar.")
        llm = ChatGroq(model=model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                       temperature=temperature, api_key=key)
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model or os.getenv("OLLAMA_MODEL", "llama3.1"), temperature=temperature)
    else:
        raise LLMConfigError(f"Unknown LLM_PROVIDER '{provider}'. Use gemini, groq or ollama.")

    return llm.with_retry(stop_after_attempt=3)
