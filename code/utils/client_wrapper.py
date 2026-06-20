import base64
import httpx
import logging
from typing import Dict, Any, List, Optional
from code.config import settings

logger = logging.getLogger(__name__)

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def handle_http_response(resp: httpx.Response, provider: str):
    if resp.status_code == 200:
        return
    if resp.status_code in [401, 403]:
        raise ValueError(
            f"Authentication failed ({resp.status_code}) on provider '{provider}': "
            f"Invalid API key or access denied. Details: {resp.text}"
        )
    elif resp.status_code == 429:
        raise RuntimeError(
            f"Rate limit exceeded (429) on provider '{provider}'. "
            f"Wait for retry backoff. Details: {resp.text}"
        )
    elif resp.status_code in [500, 502, 503, 504]:
        raise RuntimeError(
            f"Temporary server error ({resp.status_code}) on provider '{provider}'. "
            f"Retrying... Details: {resp.text}"
        )
    else:
        raise RuntimeError(
            f"API request failed with status code {resp.status_code} "
            f"on provider '{provider}'. Details: {resp.text}"
        )

async def unified_chat_completion(
    model_name: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    num_ctx: int,
    top_p: float,
    response_format: Optional[str] = None,
    images: Optional[List[str]] = None,
    ollama_client: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Unified async completions API wrapper for local Ollama, OpenAI, Gemini, Anthropic, and OpenRouter.
    """
    provider = settings.provider
    
    # 1. Local Ollama Provider
    if provider == "ollama":
        if not ollama_client:
            import ollama
            ollama_client = ollama.AsyncClient(host=settings.ollama_base_url, timeout=settings.ollama_request_timeout)
        
        ollama_messages = []
        for msg in messages:
            # Copy to avoid mutating original messages list
            ollama_messages.append(dict(msg))
            
        if images and len(ollama_messages) > 0:
            ollama_messages[-1]["images"] = images
            
        options = {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "top_p": top_p
        }
        
        chat_args = {
            "model": model_name,
            "messages": ollama_messages,
            "options": options
        }
        if response_format == "json":
            chat_args["format"] = "json"
            
        response = await ollama_client.chat(**chat_args)
        
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)
        content = response["message"]["content"]
        
        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }

    # 2. Cloud API Providers Setup
    async with httpx.AsyncClient(timeout=settings.ollama_request_timeout) as client:
        formatted_messages = []
        
        b64_images = []
        if images:
            for img_path in images:
                try:
                    b64_images.append(encode_image_to_base64(img_path))
                except Exception as e:
                    logger.error(f"Failed to encode image {img_path}: {e}")

        if provider in ["openai", "openrouter", "gemini"]:
            if provider == "openai":
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json"
                }
            elif provider == "openrouter":
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json"
                }
            else:  # gemini
                url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
                headers = {
                    "Authorization": f"Bearer {settings.gemini_api_key}",
                    "Content-Type": "application/json"
                }

            for msg in messages:
                role = msg["role"]
                body_content = msg["content"]
                
                if role == "user" and b64_images:
                    content_list = [{"type": "text", "text": body_content}]
                    for b64_img in b64_images:
                        content_list.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_img}"
                            }
                        })
                    formatted_messages.append({"role": "user", "content": content_list})
                else:
                    formatted_messages.append({"role": role, "content": body_content})

            payload = {
                "model": model_name,
                "messages": formatted_messages,
                "temperature": temperature,
                "top_p": top_p
            }
            if response_format == "json":
                payload["response_format"] = {"type": "json_object"}

            logger.info(f"Sending API request to {provider} ({model_name})")
            resp = await client.post(url, headers=headers, json=payload)
            handle_http_response(resp, provider)
                
            resp_data = resp.json()
            content = resp_data["choices"][0]["message"]["content"]
            
            usage = resp_data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            return {
                "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }

        elif provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }

            system_msg = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    role = msg["role"]
                    body_content = msg["content"]
                    
                    if role == "user" and b64_images:
                        content_list = [{"type": "text", "text": body_content}]
                        for b64_img in b64_images:
                            content_list.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64_img
                                }
                            })
                        formatted_messages.append({"role": "user", "content": content_list})
                    else:
                        # Anthropic roles are only user or assistant
                        formatted_messages.append({"role": role, "content": body_content})

            payload = {
                "model": model_name,
                "messages": formatted_messages,
                "max_tokens": 4096,
                "temperature": temperature,
                "top_p": top_p
            }
            if system_msg:
                payload["system"] = system_msg

            logger.info(f"Sending API request to anthropic ({model_name})")
            resp = await client.post(url, headers=headers, json=payload)
            handle_http_response(resp, provider)
                
            resp_data = resp.json()
            content = resp_data["content"][0]["text"]
            
            usage = resp_data.get("usage", {})
            prompt_tokens = usage.get("input_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0)
            
            return {
                "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }

        else:
            raise ValueError(f"Unsupported provider: {provider}")
