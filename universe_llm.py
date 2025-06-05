import argparse
import os
from typing import Callable, List, Dict

from langchain.prompts import PromptTemplate
from transformers import pipeline

import openai
import google.generativeai as genai


template = (
    "You are a senior astrologer. "
    "Analyze the meaning of {planet} at {sign} {degree}\u00b0"
)
prompt = PromptTemplate.from_template(template)


def _hf_generator(model_id: str, max_new_tokens: int, temperature: float) -> Callable[[str], str]:
    generator = pipeline(
        "text-generation",
        model=model_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    def _generate(text: str) -> str:
        return generator(text)[0]["generated_text"]

    return _generate


def _openai_generator(model_id: str, max_new_tokens: int, temperature: float) -> Callable[[str], str]:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _generate(text: str) -> str:
        response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=[{"role": "user", "content": text}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content

    return _generate


def _google_generator(model_id: str, max_new_tokens: int, temperature: float) -> Callable[[str], str]:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(model_id)

    def _generate(text: str) -> str:
        response = model.generate_content(
            text,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_new_tokens,
                temperature=temperature,
            ),
        )
        return response.text

    return _generate


def get_generator(provider: str, model_id: str, max_new_tokens: int, temperature: float) -> Callable[[str], str]:
    if provider == "huggingface":
        return _hf_generator(model_id, max_new_tokens, temperature)
    if provider == "openai":
        return _openai_generator(model_id, max_new_tokens, temperature)
    if provider == "google":
        return _google_generator(model_id, max_new_tokens, temperature)
    raise ValueError(f"Unknown provider: {provider}")


def generate_responses(docs: List[Dict], provider: str, model_id: str) -> List[str]:
    generator = get_generator(provider, model_id, max_new_tokens=256, temperature=0.5)
    prompts = [
        prompt.format(
            planet=doc["planet"],
            sign=doc["sign"],
            degree=round(doc["sign_degree"], 2),
        )
        for doc in docs
    ]
    return [generator(text) for text in prompts]


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM provider demo")
    parser.add_argument(
        "--provider",
        choices=["huggingface", "openai", "google"],
        default="huggingface",
        help="Which backend to use",
    )
    parser.add_argument(
        "--model-id",
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="Model identifier for the chosen provider",
    )
    args = parser.parse_args()

    example_docs = [
        {"planet": "Mars", "sign": "Aries", "sign_degree": 14.2},
        {"planet": "Venus", "sign": "Taurus", "sign_degree": 28.7},
    ]

    for output in generate_responses(example_docs, args.provider, args.model_id):
        print(output)


if __name__ == "__main__":
    main()