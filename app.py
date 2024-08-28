import os
import cv2
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
import speech_recognition as sr
from dotenv import load_dotenv
import streamlit as st
from mtcnn import MTCNN 
import google.generativeai as genai 
# import time 

load_dotenv()

path = "voice/"
os.makedirs(path, exist_ok=True)

# 1. Save and play voice created by Google Text-to-Speech (gTTS)
def text_to_audio(text, filename):
    tts = gTTS(text)
    file_path = os.path.join(path, filename)
    tts.save(file_path)
    return file_path

def play_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    play(audio)

# 2. Use microphone to record voice
def record_audio(duration=3):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        st.write(f"Recording for {duration} seconds...")

        try:
            recorded_audio = recognizer.listen(source, timeout=duration)
            st.write("Recording complete.")
            return recorded_audio
        except sr.WaitTimeoutError:
            st.write("No speech detected within the timeout period.")
            return None

# 3. Convert the recorded voice to text through speech-to-text (STT)
def audio_to_text(audio):
    recognizer = sr.Recognizer()
    try:
        st.write("Recognizing the text...")
        text = recognizer.recognize_google(audio, language="en-US", show_all=False)
        st.write("Decoded Text: {}".format(text))
    except sr.UnknownValueError:
        text = "Could not understand the audio."
    except sr.RequestError:
        text = "Request to Google STT failed."
    return text

# 4. Convert the text to voice through text-to-speech (TTS)
def text_to_speech(text):
    tts = gTTS(text)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    audio_segment = AudioSegment.from_file(audio_buffer, format="mp3")
    play(audio_segment)

# 5. Integrate an LLM to respond to voice input with voice output
genai.configure(api_key=os.getenv("GENAI_API_KEY"))
gemini_pro = genai.GenerativeModel(model_name="models/gemini-1.5-pro")

def respond_by_gemini(input_text, role_text, instructions_text):
    final_prompt = [
        "ROLE: " + role_text,
        "INPUT_TEXT: " + input_text,
        instructions_text,
    ]
    response = gemini_pro.generate_content(
        final_prompt,
        stream=True,
    )
    response_text = "".join(chunk.text for chunk in response)
    return response_text

def llm_voice_response():
    role = 'You are an intelligent assistant to chat on the topic: `{}`.'
    topic = ' '
    role_text = role.format(topic)
    instructions = 'Respond to the INPUT_TEXT briefly in chat style. Respond based on your knowledge about `{}` in brief chat style.'
    instructions_text = instructions.format(topic)

    recorded_audio = record_audio()
    if recorded_audio is None:
        st.write("No audio recorded. Please try again.")
        return "No audio recorded. Please try again.", None
    text = audio_to_text(recorded_audio)
    response_text = text
    if text not in ["Could not understand the audio.", "Request to Google STT failed."]:
        response_text = respond_by_gemini(text, role_text, instructions_text)
    text_to_speech(response_text)
    return response_text, text

# 6. Build a Web interface for the LLM-supported voice assistant
def main():
    st.set_page_config(page_title="LLM-Supported Voice Assistant", layout="wide")

    st.markdown("""
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .main {background-color: #f5f5f5;}
            .container {max-width: 800px; margin: auto; padding-top: 50px;}
            .title {font-family: 'Arial', sans-serif; color: #333333; margin-bottom: 30px;}
            .btn {background-color: #4CAF50; color: white; border: none; padding: 10px 20px; cursor: pointer; font-size: 16px;}
            .btn:hover {background-color: #45a049;}
            .chat-bubble {background-color: #e0e0e0; padding: 10px; border-radius: 10px; margin-bottom: 10px;}
            .chat-bubble.user {background-color: #52dbeb; color: white; margin-left: auto;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='container' style=text-align:center><h1 class='title'>LLM-Supported Voice Assistant</h1></div>", unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    st.write("This is a voice assistant with LLM support. The assistant will start listening when a face is detected in the webcam input.")

    stframe = st.empty()

    cap = cv2.VideoCapture(0)
    face_detector = MTCNN()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        st.error("Could not open webcam.")
        return

    response_text = ""
    user_question = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detector.detect_faces(rgb_frame)
        for result in results:
            x, y, w, h = result['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        stframe.image(frame, channels="BGR", use_column_width=False, width=400)

        if len(results) > 0:
            st.write("Face detected, starting voice assistant...")
            # start_time = time.time()
            response_text, user_question = llm_voice_response()
            # end_time = time.time()
            
            if user_question == "Could not understand the audio.":
                st.session_state.conversation.append({"role": "assistant", "text": response_text})
            else:
                st.session_state.conversation.append({"role": "user", "text": user_question})
                st.session_state.conversation.append({"role": "assistant", "text": response_text})
            
            # st.write(f"Response time: {end_time - start_time:.2f} seconds")
            break

    cap.release()

    for chat in st.session_state.conversation:
        if chat["role"] == "user":
            st.markdown(f"<div class='chat-bubble user'>{chat['role']} : {chat['text']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble'>{chat['role']}: {chat['text']}</div>", unsafe_allow_html=True)

    if response_text:
        if st.button("Listen"):
            st.session_state['retry'] = True

if __name__ == "__main__":
    main()
