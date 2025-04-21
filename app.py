import streamlit as st
import torchaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from groq import Groq

st.title("Amharic Healthcare Voice Analysis")

# Upload audio file
audio_file = st.file_uploader("Upload Amharic audio file", type=["wav", "mp3", "m4a"])
if audio_file is not None:
    # Load and process
    speech_array, sampling_rate = torchaudio.load(audio_file)
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_array = resampler(speech_array)

    # Load models
    asr_model = Wav2Vec2ForCTC.from_pretrained("tinkvu/wav2vec2-large-xlsr-amharic-healthcare")
    processor = Wav2Vec2Processor.from_pretrained("tinkvu/wav2vec2-large-xlsr-amharic-healthcare")

    inputs = processor(speech_array[0], sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = asr_model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    st.subheader("Transcription (Amharic)")
    st.write(transcription)

    # Analyze with Groq
    client = Groq()
    prompt = (
        "Analyse this Amharic transcription of a conversation between a Hospital receptionist and patient.\n"
        "From this conversation, output the Name, Age, Location and Key Symptoms of the patient in English. "
        "Add a confidence percentage on each output.\n\n"
        f"Transcription: {transcription}"
    )

    st.subheader("Extracted Patient Info (English)")
    with st.spinner("Analyzing..."):
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "system", "content": prompt}],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        result = ""
        for chunk in completion:
            result += chunk.choices[0].delta.content or ""
        st.write(result)
