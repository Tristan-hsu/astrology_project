import argparse
import os
import openai
import google.generativeai as genai 
from typing import Callable, List, Dict, Any

from langchain.prompts import PromptTemplate
from transformers import pipeline




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
            model='gpt-4o-mini',
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

def summurize(inputs:List[str],astro_data: List[Dict],provider: str, model_id: str) ->str:
    generator = get_generator(provider, model_id, max_new_tokens=4096, temperature=0.5)
    summurize_template = ("""You are a senior astrologer and your audience is Traditional Chinese Speaker. 
    The following is the content:{content},
    Summarize these astrology content into table and description of this person. 
    After that translate the content into Traditional Chinese.""")
    summurize_prompt = PromptTemplate.from_template(summurize_template)
    inputs = str(inputs)
    summurize_prompt.format(content=inputs)
    
    return generator(summurize_prompt)
    


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM provider demo")
    parser.add_argument(
        "--provider",
        choices=["huggingface", "openai", "google"],
        default="openai",
        help="Which backend to use",
    )
    parser.add_argument(
        "--model-id",
        # default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        default='gpt-4o-mini',
        help="Model identifier for the chosen provider",
    )
    args = parser.parse_args()

    docs = [{"planet":"Sun","degree":266.7796,"sign":"Sagittarius","sign_degree":26.78},
            {"planet":"Moon","degree":198.2242,"sign":"Libra","sign_degree":18.22},
            {"planet":"Mercury","degree":247.8317,"sign":"Sagittarius","sign_degree":7.83},
            {"planet":"Venus","degree":311.2985,"sign":"Aquarius","sign_degree":11.3},
            {"planet":"Mars","degree":114.9886,"sign":"Cancer","sign_degree":24.99},
            {"planet":"Jupiter","degree":192.105,"sign":"Libra","sign_degree":12.1},
            {"planet":"Saturn","degree":315.0037,"sign":"Aquarius","sign_degree":15},
            {"planet":"Uranus","degree":286.8812,"sign":"Capricorn","sign_degree":16.88},
            {"planet":"Neptune","degree":287.8558,"sign":"Capricorn","sign_degree":17.86},
            {"planet":"Pluto","degree":234.1586,"sign":"Scorpio","sign_degree":24.16}]

    for output in generate_responses(docs,args.provider,args.model_id):
        print(output)
        text_file = open("Output.txt", "w")
        text_file.write(output)
    
    _summurize = summurize(output,docs,args.provider,args.model_id)
    print(_summurize)
    text_file.write(_summurize)
    text_file.close()

if __name__ == "__main__":
    main()