"""
Astro Engine & Horoscope – Gradio Edition
========================================
A small Gradio app that exposes the same functionality as the original
FastAPI micro‑service: it returns the zodiac sign and J2000 ecliptic
longitude for the ten major solar‑system bodies for a given local
(date‑time, place) pair.

Run locally with something like::

    python astro_gradio.py               # launches at http://127.0.0.1:7860

or, from inside a notebook/colab cell::

    import astro_gradio
    astro_gradio.demo.launch(share=True)

Environment variables (optional):
    GOOGLE_TZ_API_KEY  # Google Time Zone API key, if available
    SWEPH_PATH        # Directory containing Swiss Ephemeris data files

Dependencies (same as the original service):
    pip install gradio pyswisseph python-dateutil geopy timezonefinder pytz requests
"""
