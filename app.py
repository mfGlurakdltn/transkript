import tempfile

import gradio as gr
import whisper

_modelle: dict = {}


def lade_modell(name: str):
    if name not in _modelle:
        _modelle[name] = whisper.load_model(name)
    return _modelle[name]


def transkribiere(audio_datei, modell_name: str):
    if audio_datei is None:
        return "", None

    modell = lade_modell(modell_name)
    ergebnis = modell.transcribe(audio_datei, language="de", verbose=False)
    text = ergebnis["text"].strip()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(text)
        temp_pfad = f.name

    return text, temp_pfad


with gr.Blocks(title="Audio Transkription 🎙️") as demo:
    gr.Markdown("# 🎙️ Audio Transkription")
    gr.Markdown(
        "Lade eine Audiodatei hoch (mp3, wav, m4a, ...) und erhalte eine deutsche Transkription."
    )

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Audiodatei")
        modell_auswahl = gr.Dropdown(
            choices=["small", "medium", "large"],
            value="medium",
            label="Modell (large = genauer, langsamer)",
        )

    btn = gr.Button("Transkribieren ▶", variant="primary", size="lg")

    text_output = gr.Textbox(
        label="Transkription", lines=12, show_copy_button=True, interactive=False
    )
    datei_output = gr.File(label="Als .txt herunterladen")

    btn.click(
        fn=transkribiere,
        inputs=[audio_input, modell_auswahl],
        outputs=[text_output, datei_output],
    )

demo.launch()
