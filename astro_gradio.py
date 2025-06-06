
from __future__ import annotations


import datetime as dt
from astro_computation import astro_data
from universe_llm import generate_responses,summurize

import gradio as gr
# ---------------------------------------------------------------------------
# Swiss Ephemeris initialisation – call once at import time
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Gradio Interface definition
# ---------------------------------------------------------------------------

def _build_interface() -> gr.Blocks:
    with gr.Blocks(title="Horoscope Astro Engine") as demo:
        gr.Markdown("## Horoscope Astro Engine – Planetary Positions")
        with gr.Row():
            datetime_input = gr.Textbox(
                label="Date & Time (ISO‑8601)",
                value=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            )
            location_input = gr.Textbox(label="Location", value="Taipei")
            llm_provider = gr.Dropdown(["huggingface", "openai", "google"],
            label="請選擇一個選項"
            )
    
            api_key = gr.Textbox(
            label="API-key",
            type="password",
            placeholder="API-key"
            )
            model_id = gr.Textbox(
                label="Model",
                value="gpt-4o-mini"
            )
        run_btn = gr.Button("Calculate")
        output= gr.Textbox(label="Ephemeris")

        
        def _on_click(dt_str: str, loc: str):
            try:
                docs = astro_data(dt_str, loc)
                responses = generate_responses(docs, llm_provider, model_id, api_key)
                _summurize =summurize(responses,llm_provider, model_id, api_key)
                return _summurize
            except Exception as exc:
                return {"error": str(exc)}

        run_btn.click(_on_click, inputs=[datetime_input, location_input], outputs=output)

    return demo

# Instantiate the interface at import time so that external callers can do
# `import astro_gradio; astro_gradio.demo.launch()`

demo: gr.Blocks = _build_interface()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch()