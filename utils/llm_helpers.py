import os
import requests
from utils.llm_config import LLM_PRIORITY # Make sure this file exists and is structured as above

def call_groq(prompt, model, max_tokens, temperature):
    groq_api_key = os.getenv("GROQ_API_KEY")
    # <<< SUGGESTION: Proactive API Key Check >>>
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
        
    # <<< CRITICAL FIX: Corrected URL >>>
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(url, headers=headers, json=body, timeout=60)
    resp.raise_for_status() # Will raise an exception for 4xx/5xx errors
    result = resp.json()
    return result["choices"][0]["message"]["content"]

def call_gemini(prompt, model, max_tokens, temperature):
    google_api_key = os.getenv("GOOGLE_API_KEY")
    # <<< SUGGESTION: Proactive API Key Check >>>
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={google_api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}
    }
    resp = requests.post(url, headers=headers, json=data, timeout=60)
    resp.raise_for_status() # Will raise an exception for 4xx/5xx errors
    result = resp.json()
    return result["candidates"][0]["content"]["parts"][0]["text"]

PROVIDER_FN = {
    "groq": call_groq,
    "google": call_gemini,
}

def query_llm(prompt, provider=None, model=None, max_tokens=512, temperature=0.7, **kwargs):
    # This function's logic is already excellent and needs no changes.
    tried = set()
    errors = []
    
    # Build the list of configurations to try, prioritizing the user's explicit choice
    config_list = []
    if provider and model:
        config_list.append({"provider": provider, "model": model})
    
    # Add the rest from the priority list, avoiding duplicates
    for p_conf in LLM_PRIORITY:
        if (p_conf["provider"], p_conf["model"]) not in [(provider, model)]:
            config_list.append(p_conf)

    # Iterate through the final configuration list and try each one
    for conf in config_list:
        prov = conf["provider"]
        mod = conf["model"]
        fn = PROVIDER_FN.get(prov)
        
        if not fn or (prov, mod) in tried:
            continue
        
        tried.add((prov, mod))
        
        try:
            print(f"Attempting to call {prov.upper()} with model {mod}...")
            return fn(prompt, model=mod, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            print(f"FAILED to call {prov.upper()}:{mod}. Reason: {e}")
            errors.append(f"{prov}:{mod} -- {e}")

    # If all providers in the list have failed
    raise RuntimeError("All LLM providers failed. Errors: " + " | ".join(str(e) for e in errors))