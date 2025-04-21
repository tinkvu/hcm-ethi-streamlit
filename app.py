import streamlit as st
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from groq import Groq
import io
import os

# Groq API setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

st.title("Amharic Medical Voice Analysis")

uploaded_file = st.file_uploader("Upload Amharic audio file", type=["wav", "mp3", "m4a"])
if uploaded_file is not None:
    # Read audio file using soundfile
    audio_bytes = uploaded_file.read()
    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))

    # Resample if not 16kHz
    if sample_rate != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    # Load model and processor
    processor = Wav2Vec2Processor.from_pretrained("tinkvu/wav2vec2-large-xlsr-amharic-healthcare")
    model = Wav2Vec2ForCTC.from_pretrained("tinkvu/wav2vec2-large-xlsr-amharic-healthcare")

    # Process and transcribe
    inputs = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    st.subheader("🗣️ Transcription (Amharic)")
    st.write(transcription)

    # Groq prompt
    prompt = (
        "Analyse this Amharic transcription of a conversation between a Hospital receptionist and patient.\n"
        "From this conversation, output the Name, Age, Location and Key Symptoms of the patient in English. "
        "Add a confidence percentage on each output.\n\n"
        f"Transcription: {transcription}"
    )

    st.subheader("📋 Extracted Information (English)")
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
