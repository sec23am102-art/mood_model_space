# app.py  (for Hugging Face Space)
import gradio as gr
from transformers import pipeline

# load model (will download on first build/start)
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False
)

mapping = {
    'joy': 'happy','amusement': 'happy','excitement': 'happy',
    'pride': 'motivated','desire': 'night','love': 'motivated',
    'gratitude': 'happy','sadness': 'sad','disappointment': 'sad',
    'nervousness': 'night','anger': 'angry','annoyance': 'angry',
    'embarrassment': 'night','neutral': 'neutral','curiosity': 'motivated',
    'realization': 'motivated','sarcasm': 'angry','confusion': 'neutral',
    'fear': 'night'
}

def predict(text):
    # Expecting a single string and returning mapped label
    if not text:
        return "neutral"
    result = emotion_classifier(text)[0]
    label = result.get("label", "").lower()
    return mapping.get(label, "neutral")

# Expose a tiny UI (optional) and an API endpoint
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, placeholder="Type something..."),
    outputs="text",
    title="MoodTuneBot Emotion API",
    description="Returns mapped mood label (happy/sad/angry/...) for a given text."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
