from __future__ import annotations

import datetime as dt

import gradio as gr

from astro_computation import astro_data
from universe_llm import generate_responses, summurize


def summarize_astrology(
    datetime_str: str,
    location: str,
    provider: str,
    api_key: str,
    model_id: str,
) -> str:
    """Generate a Chinese horoscope summary for the given parameters."""
    try:
        docs = astro_data(datetime_str, location)
        print(123)
        responses = generate_responses(docs, provider, model_id, api_key)
        print(456)
        return summurize(responses, provider, model_id, api_key)
    except Exception as exc:
        return f"Error: {exc}"


demo = gr.Interface(
    summarize_astrology,
    [
        gr.Textbox(
            label="Date & Time (ISO‑8601)",
            value=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        ),
        gr.Textbox(label="Location", value="Taipei"),
        gr.Dropdown(
            ["huggingface", "openai", "google"],
            label="請選擇一個選項",
        ),
        gr.Textbox(label="API-key", type="password", placeholder="API-key"),
        gr.Textbox(label="Model", value="gpt-4o-mini"),
    ],
    gr.Textbox(label="Ephemeris"),
    title="Horoscope Astro Engine",
    description="Horoscope Astro Engine – Planetary Positions",
    submit_btn="Calculate",
)


if __name__ == "__main__":
    demo.launch()
