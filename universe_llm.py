import argparse
import os
import openai
import google.generativeai as genai 
from typing import Callable, List, Dict, Any

from langchain.prompts import PromptTemplate
from transformers import pipeline




template = (
    "You are a senior astrologer. "
    "Analyze the meaning of {planet} at {sign} {degree}, with speed {speed} and retrogarde state {retrograde}\u00b0"
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


def _openai_generator(model_id: str, max_new_tokens: int, temperature: float,api_key: str = None) -> Callable[[str], str]:
    if api_key:
        client = openai.OpenAI(api_key=api_key)
    else:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _generate(text: str) -> str:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": text}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content

    return _generate


def _google_generator(model_id: str, max_new_tokens: int, temperature: float,api_key: str = None) -> Callable[[str], str]:
    if api_key:
        genai.configure(api_key=api_key)
    else:
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


def get_generator(provider: str, model_id: str, max_new_tokens: int, temperature: float, api_key: str = None) -> Callable[[str], str]:
    if provider == "huggingface":
        return _hf_generator(model_id, max_new_tokens, temperature,api_key)
    if provider == "openai":
        return _openai_generator(model_id, max_new_tokens, temperature,api_key)
    if provider == "google":
        return _google_generator(model_id, max_new_tokens, temperature,api_key)
    raise ValueError(f"Unknown provider: {provider}")


def generate_responses(docs: List[Dict], provider: str, model_id: str,api_key=None) -> List[str]:
    generator = get_generator(provider, model_id, max_new_tokens=256, temperature=0.5, api_key=None)
    prompts = [
        prompt.format(
            planet=doc["planet"],
            sign=doc["sign"],
            degree=round(doc["sign_degree"], 2),
            speed = doc["speed"],
            retrograde = doc["retrograde"]
        )
        for doc in docs
    ]
    return [generator(text) for text in prompts]

def summurize(inputs:List[str],provider: str, model_id: str,api_key=None) ->str:
    generator = get_generator(provider, model_id, max_new_tokens=4096, temperature=0.5, api_key=None)

    
    summarize_template = ("""You are a senior astrologer. 
    The following is the content:{content},
    Summarize these astrology content into paragraph and description of this person. 
    """)
    prompt_summarize_template = PromptTemplate(template=summarize_template, input_variables=['content'])
    

    translate_template = ("""Your audience is Traditional Chinese Speaker and you are senior translator of Traditional Chinese(繁體中文) and English. 
                          Translate the following content into Traditional Chinese(繁體中文).
                           {english_summary}""")
    
    prompt_translate_template= PromptTemplate(template=translate_template, input_variables=['english_summary'])

    # summurize chain 
    content = "\n".join(inputs)

    summarize_prompt = prompt_summarize_template.format(content=content)
    english_summary = generator(summarize_prompt)

    translate_prompt = prompt_translate_template.format(english_summary=english_summary)
    chinese_summary = generator(translate_prompt)

    return chinese_summary
    


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

    docs = [{"planet":"Sun","degree":266.7796,"sign":"Sagittarius","sign_degree":26.78,"speed":1.0181069731560166,"retrograde":False},
        {"planet":"Moon","degree":198.2242,"sign":"Libra","sign_degree":18.22,"speed":13.869260023596233,"retrograde":False},
        {"planet":"Mercury","degree":247.8317,"sign":"Sagittarius","sign_degree":7.83,"speed":1.3664351205868346,"retrograde":False},
        {"planet":"Venus","degree":311.2985,"sign":"Aquarius","sign_degree":11.3,"speed":1.1535253056349868,"retrograde":False},
        {"planet":"Mars","degree":114.9886,"sign":"Cancer","sign_degree":24.99,"speed":-0.2677003648667485,"retrograde":True},
        {"planet":"Jupiter","degree":192.105,"sign":"Libra","sign_degree":12.1,"speed":0.11858796055508439,"retrograde":False},
        {"planet":"Saturn","degree":315.0037,"sign":"Aquarius","sign_degree":15,"speed":0.09299917754771617,"retrograde":False},
        {"planet":"Uranus","degree":286.8812,"sign":"Capricorn","sign_degree":16.88,"speed":0.05674487346380018,"retrograde":False},
        {"planet":"Neptune","degree":287.8558,"sign":"Capricorn","sign_degree":17.86,"speed":0.03585330432602562,"retrograde":False},
        {"planet":"Pluto","degree":234.1586,"sign":"Scorpio","sign_degree":24.16,"speed":0.03525563243877108,"retrograde":False}]
    text_file = open("Output.txt", "a")
    responses = generate_responses(docs,args.provider,args.model_id)
    for output in responses:
        print(output)
        text_file.write(output)

    
    _summurize =summurize(responses,args.provider,args.model_id)
    print(_summurize)
    text_file.write(_summurize)
    text_file.close()

if __name__ == "__main__":
    main()
