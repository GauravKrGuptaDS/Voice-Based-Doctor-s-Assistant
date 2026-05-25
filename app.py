import streamlit as st
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI
import tempfile

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
    #api_key=""
)

st.title("🎙 Voice-Based Clinic Assistant")

st.write("Speak your clinic command")

audio = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    key='recorder'
)

def transcribe_audio(audio_bytes):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(audio_bytes)
        temp_path = tmp.name

    with open(temp_path, "rb") as audio_file:

        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

    return transcript.text


def understand_command(command):

    prompt = f"""
    Understand the doctor's intent.

    Possible actions:
    - show appointments
    - book follow-up
    - patient summary
    - reminders

    Doctor command:
    {command}

    Return action only.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


if audio:

    st.audio(audio["bytes"])

    transcript = transcribe_audio(
        audio["bytes"]
    )

    st.success(f"You said: {transcript}")

    action = understand_command(
        transcript
    )

    st.subheader("AI Understanding")

    st.write(action)

    # Mock actions
    if "show appointments" in action.lower():

        st.subheader(
            "Today's Appointments"
        )

        st.write("""
        10:00 AM - Rahul Sharma
        11:30 AM - Priya Gupta
        2:00 PM - Amit Jain
        """)

    elif "follow-up" in action.lower():

        st.success(
            "Follow-up booked after 7 days"
        )
