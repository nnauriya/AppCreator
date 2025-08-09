import requests
import os

def call_groq(prompt, model, max_tokens, temperature, **kwargs):
    """
    Calls the Groq API. Fetches the API key from the environment.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not found.")

    # <<< CRITICAL FIX: The URL now includes the /openai/ path segment >>>
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    response = requests.post(url, headers=headers, json=data, timeout=60)
    response.raise_for_status() # Raises an exception for HTTP error codes
    content = response.json()
    
    # Your safe parsing logic is great and remains unchanged
    return content["choices"][0]["message"]["content"] if ("choices" in content and content["choices"]) else None

def call_gemini(prompt, model, max_tokens, temperature, **kwargs):
    """
    Calls the Google Gemini API. Fetches the API key from the environment for consistency.
    """
    # <<< SUGGESTION: Fetch key from environment to be consistent with call_groq >>>
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not found.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={google_api_key}"
    
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature
        }
    }
    response = requests.post(url, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    content = response.json()

    # Your safe parsing logic is great and remains unchanged
    return content["candidates"][0]["content"]["parts"][0]["text"] if ("candidates" in content and content["candidates"]) else None