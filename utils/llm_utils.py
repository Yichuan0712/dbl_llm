from typing import Optional, List, Dict, Callable
from extractor.request_openai import (
    # request_to_chatgpt_35,
    # request_to_chatgpt_40,
    request_to_chatgpt_4o,
)
from extractor.request_geminiai import (
    request_to_gemini_15_pro,
    request_to_gemini_15_flash,
)
import ast
# from dotenv import load_dotenv
# load_dotenv()


def get_llm_response(messages, question, model="gemini_15_pro"):
    """
    A further wrapper around Shaohong's request_llm function.
    Send messages and question to the specified LLM and return response details.
    """

    prompt_list = [{"role": "user", "content": msg} for msg in messages]

    if model == "chatgpt_4o":
        request_llm = request_to_chatgpt_4o
    elif model == "gemini_15_pro":
        request_llm = request_to_gemini_15_pro
    else:
        raise ValueError(f"Unsupported model: {model}")

    res, content, usage, truncated = request_llm(prompt_list, question)
    return res, content, usage, truncated


def fix_angle_brackets(text: str) -> str:
    text = text.rstrip()
    return text + '>' if text.endswith('>') and not text.endswith('>>') else text


def fix_trailing_brackets(text):
    try:
        ast.literal_eval(text)
        return text
    except Exception:
        pass
    for bracket in ["]", ")", "}"]:
        try:
            if ast.literal_eval(text + bracket):
                return text + bracket
        except Exception:
            continue
    return text


