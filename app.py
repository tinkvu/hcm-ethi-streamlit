import streamlit as st
import torchaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from groq import Groq
import io
import os

# Ensure event loop for Streamlit
import asyncio
if not asyncio.get_event_loop().is_running():
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load Groq API key from Streamlit secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Streamlit UI
st.title("Amharic Healthcare Voice Analysis")
audio_file = st.file_uploader("Upload Amharic audio file", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    # Read audio bytes
    audio_bytes = audio_file.read()
    audio_tensor, sampling_rate = torchaudio.load(io.BytesIO(audio_bytes))  # Use buffer here

    # Resample if needed
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        audio_tensor = resampler(audio_tensor)

    # Load model and processor
    model = Wav2Vec2ForCTC.from_pretrained("tinkvu/wav2vec2-large-xlsr-amharic-healthcare")
    processor = Wav2Vec2Processor.from_pretrained("tinkvu/wav2vec2-large-xlsr-amharic-healthcare")

    # Run transcription
    inputs = processor(audio_tensor[0], sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    st.subheader("Transcription (Amharic)")
    st.write(transcription)

    # Groq prompt
    prompt = (
        "Analyse this Amharic transcription of a conversation between a Hospital receptionist and patient.\n"
        "From this conversation, output the Name, Age, Location and Key Symptoms of the patient in English. "
        "Add a confidence percentage on each output.\n\n"
        f"Transcription: {transcription}"
    )

    st.subheader("Extracted Info (English)")
    with st.spinner("Analyzing..."):
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "system", "content": prompt}],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
        )
        result = ""
        for chunk in completion:
            result += chunk.choices[0].delta.content or ""
        st.write(result)
