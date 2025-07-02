import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import json
import time
import tempfile
import os
import asyncio
import edge_tts
from faster_whisper import WhisperModel
import base64
import streamlit_mic_recorder as st_mic_recorder
import nest_asyncio
import threading
import concurrent.futures

# Apply nest_asyncio to handle nested event loops
nest_asyncio.apply()

# --- Page config must be first ---
st.set_page_config(page_title="AI Tutor", page_icon="🎓", layout="wide")

# --- Initialize embedding model ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# --- Initialize Whisper model for STT ---
@st.cache_resource
def load_whisper_model():
    return WhisperModel("tiny", compute_type="int8")

whisper_model = load_whisper_model()

# --- Mistral API setup ---
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
# Ensure MISTRAL_API_KEY is in Streamlit secrets
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {MISTRAL_API_KEY}",
    "Content-Type": "application/json"
}

# --- Audio to Text Function ---
def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio_bytes)
        audio_path = tmpfile.name

    segments, _ = whisper_model.transcribe(audio_path)
    full_text = " ".join([s.text for s in segments])
    os.remove(audio_path)  # Clean up temp file
    return full_text

# --- FIXED Text to Audio Functions ---
def text_to_speech_sync(text, voice="en-US-JennyNeural"):
    """
    Synchronous wrapper for Edge-TTS that works with Streamlit
    """
    try:
        # Create a new event loop for this function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Run the async function
        audio_bytes = loop.run_until_complete(text_to_speech_async(text, voice))
        loop.close()
        # Check for valid MP3 header and non-empty
        if (
            not audio_bytes or len(audio_bytes) < 4 or
            not (
                audio_bytes.startswith(b'\x49\x44\x33') or  # ID3
                audio_bytes.startswith(b'\xff\xf3') or     # MPEG1 Layer III
                audio_bytes.startswith(b'\xff\xfb')        # MPEG1 Layer III (alt)
            )
        ):
            st.error("Edge-TTS did not return a valid MP3 audio file.")
            return None
        return audio_bytes
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")
        return None

async def text_to_speech_async(text, voice="en-US-JennyNeural"):
    """
    Async function to generate speech using Edge-TTS
    """
    try:
        # Create communication object
        communicate = edge_tts.Communicate(text, voice)
        
        # Use temporary file approach (more reliable)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        # Save audio to temporary file
        await communicate.save(temp_path)
        
        # Read the file
        with open(temp_path, 'rb') as f:
            audio_bytes = f.read()
        
        # Clean up
        os.unlink(temp_path)
        
        return audio_bytes
        
    except Exception as e:
        print(f"Edge-TTS async error: {e}")
        return None

def text_to_speech_threaded(text, voice="en-US-JennyNeural"):
    """
    Thread-based approach to avoid event loop conflicts, with extra diagnostics.
    """
    def run_tts():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            async def generate_audio():
                communicate = edge_tts.Communicate(text, voice)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                    temp_path = tmp_file.name
                await communicate.save(temp_path)
                # Ensure file is closed before reading
                with open(temp_path, 'rb') as f:
                    audio_data = f.read()
                # Diagnostic: print file size and first bytes
                print(f"[EdgeTTS] Text: {text}")
                print(f"[EdgeTTS] File size: {len(audio_data)} bytes")
                print(f"[EdgeTTS] First 8 bytes: {audio_data[:8]}")
                os.unlink(temp_path)
                # Check for valid MP3 header and non-empty
                if (
                    not audio_data or len(audio_data) < 4 or
                    not (
                        audio_data.startswith(b'\x49\x44\x33') or  # ID3
                        audio_data.startswith(b'\xff\xf3') or     # MPEG1 Layer III
                        audio_data.startswith(b'\xff\xfb')        # MPEG1 Layer III (alt)
                    )
                ):
                    print(f"[EdgeTTS] Invalid MP3 header or empty file: {audio_data[:16]}")
                    return None
                return audio_data
            result = loop.run_until_complete(generate_audio())
            loop.close()
            return result
        except Exception as e:
            print(f"Threaded TTS error: {e}")
            return None
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_tts)
        try:
            audio_bytes = future.result(timeout=30)  # 30 second timeout
            if not audio_bytes:
                st.error("Edge-TTS did not return a valid MP3 audio file.")
            return audio_bytes
        except concurrent.futures.TimeoutError:
            st.error("Audio generation timed out")
            return None
        except Exception as e:
            st.error(f"Audio generation failed: {e}")
            return None

# --- Test Function for Sidebar ---
def test_edge_tts():
    """Test function for the sidebar"""
    try:
        test_text = "This is a test of the Edge TTS system. If you can hear this, the audio is working correctly."
        
        with st.spinner("Testing Edge-TTS..."):
            # Try threaded approach
            audio_bytes = text_to_speech_threaded(test_text)
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
                st.download_button(
                    label="Download Test MP3",
                    data=audio_bytes,
                    file_name="edge_tts_test.mp3",
                    mime="audio/mp3"
                )
                st.success("✅ Edge-TTS test successful!")
                return True
            else:
                # Try sync approach as fallback
                audio_bytes = text_to_speech_sync(test_text)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                    st.success("✅ Edge-TTS test successful (fallback method)!")
                    return True
                else:
                    st.error("❌ Edge-TTS test failed - no audio generated")
                    return False
                    
    except Exception as e:
        st.error(f"❌ Edge-TTS test failed: {e}")
        return False

# --- PDF Parsing ---
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --- Text Chunking ---
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# --- Embedding & Retrieval ---
def embed_chunks(chunks):
    embeddings = embedder.encode(chunks)
    return np.array(embeddings)

def get_top_k_chunks(query, chunks, chunk_embeddings, k=5):
    query_embedding = embedder.encode([query])[0]
    similarities = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    top_k_idx = similarities.argsort()[-k:][::-1]
    return [chunks[i] for i in top_k_idx], similarities[top_k_idx]

# --- Enhanced Detection Functions ---
def is_greeting(user_input):
    """Detect if the input is a simple greeting"""
    greeting_words = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings"]
    input_lower = user_input.lower().strip()
    return (any(word in input_lower for word in greeting_words) and
            len(user_input.split()) <= 3)

def is_confusion_expression(user_input):
    """Detect expressions of confusion or uncertainty"""
    confusion_phrases = [
        "i don't know", "i dont know", "don't know", "not sure", "i'm not sure",
        "no idea", "unsure", "i don't understand", "don't understand",
        "confused", "help me", "explain", "i have no idea",
        "no clue", "clueless", "lost", "i'm lost", "help", "hmmm", "im lost", "im not sure"
    ]
    input_lower = user_input.lower().strip()
    return any(phrase in input_lower for phrase in confusion_phrases)

def has_educational_context(user_input, previous_context=None, chunks=None, chunk_embeddings=None, similarity_threshold=0.15):
    """
    Check if the confusion is in context of educational material using semantic similarity.
    """
    # If there's previous context from the conversation, it's likely educational
    if previous_context:
        return True, 1.0

    # If we don't have PDF content loaded, we can't determine context
    if chunks is None or chunk_embeddings is None:
        return False, 0.0

    # Use semantic similarity to check if the confusion relates to course content
    try:
        # Get embedding for the user input
        input_embedding = embedder.encode([user_input])[0]

        # Calculate similarities with all chunks
        similarities = np.dot(chunk_embeddings, input_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(input_embedding)
        )

        max_similarity = np.max(similarities)

        # If the confusion has reasonable similarity to course content, it's educational
        is_educational = max_similarity >= similarity_threshold

        return is_educational, max_similarity

    except Exception as e:
        # If there's any error in similarity calculation, default to non-educational
        st.error(f"Error in educational context detection: {e}")
        return False, 0.0

def get_confusion_context_chunks(user_input, chunks, chunk_embeddings, k=3):
    """
    Get the most relevant chunks for confusion context, specifically for supportive responses.
    """
    if chunks is None or chunk_embeddings is None:
        return [], []

    try:
        input_embedding = embedder.encode([user_input])[0]
        similarities = np.dot(chunk_embeddings, input_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(input_embedding)
        )
        top_k_idx = similarities.argsort()[-k:][::-1]
        return [chunks[i] for i in top_k_idx], similarities[top_k_idx]
    except Exception as e:
        st.error(f"Error getting confusion context chunks: {e}")
        return [], []

# --- Mistral Streaming Response ---
def generate_mistral_response(context, user_question, chat_history, response_type="normal"):
    # Build conversation history for context
    if response_type == "confusion_with_context":
        system_content = f"""You are a supportive AI tutor helping a student who has expressed confusion or uncertainty. The student needs encouragement and gentle guidance.

CRITICAL RULES:
1. The student has indicated they don't know something or are confused - this is NORMAL and part of learning
2. Provide hints, clues, and encouragement to guide them toward understanding
3. Break down complex topics into smaller, manageable pieces
4. Use the course content to provide specific guidance and examples
5. Be patient, supportive, and encouraging
6. Ask guiding questions to help them think through the problem
7. Never make them feel bad for not knowing - confusion is part of learning!

Course Content (most relevant to their confusion):
{context}

The student is expressing uncertainty about something related to the course material - help them learn step by step."""
    else:
        system_content = f"""You are a strict AI tutor that ONLY uses the provided course content to help students.

CRITICAL RULES:
1. You can ONLY discuss topics that are directly mentioned in the provided course content
2. If a question is outside the scope of the uploaded content, respond warmly but redirect: "I'd love to help you with that! However, I can only assist with topics that are covered in your uploaded course material."
3. Use the provided course content to guide students with hints and questions - do NOT give direct answers
4. Be encouraging and supportive, but stay strictly within the bounds of the uploaded material
5. Never hallucinate or make up information not present in the course content

Course Content:
{context}"""

    messages = [{"role": "system", "content": system_content}]

    # Add chat history
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current question
    messages.append({"role": "user", "content": user_question})

    payload = {
        "model": "mistral-medium",
        "messages": messages,
        "temperature": 0.3,
        "stream": True
    }

    full_response_content = ""
    try:
        response = requests.post(MISTRAL_API_URL, headers=HEADERS, json=payload, stream=True)
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[6:]
                        if line.strip() == '[DONE]':
                            break
                        try:
                            json_data = json.loads(line)
                            if 'choices' in json_data and len(json_data['choices']) > 0:
                                delta = json_data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    full_response_content += delta['content']
                                    yield full_response_content + "▌"
                        except json.JSONDecodeError:
                            continue
            yield full_response_content
        else:
            st.error(f"Mistral API error: {response.status_code} - {response.text}")
            yield "I apologize, but I'm having trouble processing your request right now. Please try again."
    except Exception as e:
        st.error(f"Error connecting to Mistral API: {e}")
        yield "I apologize, but I'm having trouble processing your request right now. Please try again."

# --- Chat History Management ---
def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "last_topic_context" not in st.session_state:
        st.session_state.last_topic_context = None
    if "voice_mode" not in st.session_state:
        st.session_state.voice_mode = False
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    if "audio_input_key" not in st.session_state:
        st.session_state.audio_input_key = 0

def add_to_chat_history(role, content, audio_bytes=None):
    st.session_state.chat_history.append({"role": role, "content": content, "audio_bytes": audio_bytes})

def display_chat_history():
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                if message.get("audio_bytes"):
                    st.audio(message["audio_bytes"], format="audio/mp3")

def get_conversation_context():
    """Get recent conversation context to understand if confusion is in educational context"""
    if len(st.session_state.chat_history) >= 2:
        recent_messages = st.session_state.chat_history[-4:]
        recent_text = " ".join([msg["content"] for msg in recent_messages])
        return recent_text
    return None

# --- FIXED Process user input function ---
def process_user_input_sync(user_input, chunks, embeddings, similarity_threshold):
    """Updated version with fixed audio handling"""
    is_greeting_input = is_greeting(user_input)
    is_confusion_input = is_confusion_expression(user_input)
    conversation_context = get_conversation_context()

    has_edu_context, edu_similarity_score = has_educational_context(
        user_input,
        conversation_context,
        chunks,
        embeddings,
        similarity_threshold
    )

    top_chunks, similarities = get_top_k_chunks(
        user_input,
        chunks,
        embeddings
    )
    max_similarity = max(similarities) if len(similarities) > 0 else 0

    response_text = ""
    audio_bytes = None

    if is_greeting_input:
        response_text = "Hello! I'm excited to help you explore and understand your course material. I'm here to guide you through the content with questions and hints to help you learn effectively. What topic from your uploaded material would you like to dive into?"
        
        with st.chat_message("assistant"):
            st.write(response_text)

    elif is_confusion_input and has_edu_context:
        confusion_chunks, confusion_similarities = get_confusion_context_chunks(
            user_input,
            chunks,
            embeddings
        )

        context = "\n\n".join(confusion_chunks) if confusion_chunks else "\n\n".join(top_chunks)
        st.session_state.last_topic_context = context

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk_of_text in generate_mistral_response(
                context,
                user_input,
                st.session_state.chat_history[:-1],
                response_type="confusion_with_context"
            ):
                full_response = chunk_of_text.replace("▌", "")
                message_placeholder.write(chunk_of_text)
                time.sleep(0.01)
            response_text = full_response
            message_placeholder.write(response_text)

    elif max_similarity >= 0.1:
        context = "\n\n".join(top_chunks)
        st.session_state.last_topic_context = context

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk_of_text in generate_mistral_response(
                context,
                user_input,
                st.session_state.chat_history[:-1]
            ):
                full_response = chunk_of_text.replace("▌", "")
                message_placeholder.write(chunk_of_text)
                time.sleep(0.01)
            response_text = full_response
            message_placeholder.write(response_text)

    else:
        if is_confusion_input:
            response_text = f"I understand you're feeling uncertain, but this seems to be outside the scope of your uploaded course material. I can only help with topics covered in your PDF. Feel free to ask me about any concepts, theories, or topics from your course material!"
        else:
            response_text = "I'd love to help you with that! However, I can only assist with topics that are covered in your uploaded course material. This question seems to be outside the scope of what we've covered together. What would you like to explore from your course content?"

        with st.chat_message("assistant"):
            st.write(response_text)

    # FIXED: Handle audio generation if in voice mode
    if st.session_state.voice_mode and response_text:
        try:
            with st.spinner("Generating audio response..."):
                audio_bytes = text_to_speech_threaded(response_text)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                    st.download_button(
                        label="Download Response MP3",
                        data=audio_bytes,
                        file_name="ai_tutor_response.mp3",
                        mime="audio/mp3"
                    )
                else:
                    st.warning("Failed to generate audio response.")
        except Exception as e:
            st.warning(f"Audio error: {e}")

    # Add assistant response to chat history
    add_to_chat_history("assistant", response_text, audio_bytes)

# --- Main Streamlit UI ---
st.title("🎓 AI Tutor - Learn with Guidance")

# Initialize session state
initialize_session_state()

# Sidebar for PDF upload
with st.sidebar:
    st.header("📚 Course Material")
    uploaded_file = st.file_uploader("Upload your course PDF", type=["pdf"])

    # Fixed similarity threshold
    similarity_threshold = 0.15

    if uploaded_file:
        if st.session_state.chunks is None or st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            with st.spinner("Processing your PDF..."):
                full_text = extract_text_from_pdf(uploaded_file)
                st.session_state.chunks = chunk_text(full_text)
                st.session_state.embeddings = embed_chunks(st.session_state.chunks)
            st.success("✅ PDF processed successfully!")
        else:
            st.success("✅ PDF ready!")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.last_topic_context = None
        st.rerun()

    st.subheader("🗣️ Interface Mode")
    st.session_state.voice_mode = st.toggle("Enable Voice Mode", value=st.session_state.voice_mode,
                                            help="Toggle between typing and speaking to the tutor.")

    # Debug section
    with st.expander("🔧 Debug Info"):
        st.write(f"Voice mode: {st.session_state.voice_mode}")
        st.write(f"Chat history length: {len(st.session_state.chat_history)}")
        if st.button("Test EdgeTTS"):
            test_edge_tts()

# Main chat interface
if uploaded_file and st.session_state.chunks is not None:
    st.markdown("### 💬 Chat with your AI Tutor")

    # Display chat history
    display_chat_history()

    if st.session_state.voice_mode:
        st.info("🎙️ Speak your question after pressing 'Record'.")
        
        audio_input = st_mic_recorder.mic_recorder(
            start_prompt="🔴 Record", 
            stop_prompt="⏹ Stop", 
            key=f"voice_recorder_{st.session_state.audio_input_key}"
        )
        
        if audio_input:
            with st.spinner("Transcribing audio..."):
                user_input = transcribe_audio(audio_input["bytes"])
            st.write(f"**You said:** {user_input}")

            with st.chat_message("user"):
                st.write(user_input)

            add_to_chat_history("user", user_input)
            process_user_input_sync(user_input, st.session_state.chunks, st.session_state.embeddings, similarity_threshold)
            
            st.session_state.audio_input_key += 1
            st.rerun()

    else:
        user_input = st.chat_input("Ask me anything about your course material...")
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)

            add_to_chat_history("user", user_input)
            process_user_input_sync(user_input, st.session_state.chunks, st.session_state.embeddings, similarity_threshold)

else:
    st.info("👆 Please upload a course PDF in the sidebar to begin learning!")
    st.markdown("""
    ### How to use this Enhanced AI Tutor:

    1.  **Upload your course material** (PDF) using the sidebar
    2.  **Ask questions** about the content in the chat
    3.  **Express confusion freely** - the tutor uses AI to understand if your confusion relates to the course material
    4.  **Get personalized guidance** - the tutor provides targeted help based on the most relevant parts of your material
    5.  **Toggle Voice Mode** - Use the toggle in the sidebar to switch between typing and speaking your questions, and hearing the tutor's responses!

    **New Features:**
    -   ✨ **Smart Context Detection**: Uses AI embeddings instead of hardcoded keywords
    -   🎯 **Targeted Confusion Support**: Finds the most relevant content for your specific confusion
    -   📚 **Universal Compatibility**: Works with any subject matter or course content
    -   🎙️ **Voice Input**: Speak your questions to the tutor
    -   🗣️ **Voice Output (Edge-TTS)**: Hear the tutor's responses with improved naturalness

    **Note:** Your chat history is session-based and will be cleared when you close the browser.
    """)