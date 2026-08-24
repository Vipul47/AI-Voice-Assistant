import hashlib
import html
import io

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from google import genai

from config import MAX_OUTPUT_TOKENS, TEXT_MODEL, THINKING_LEVEL, TTS_MODEL, TTS_VOICE, get_api_key
from gemini_service import fetch_ai_response, text_to_audio, transcribe_audio

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AudioFlow",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

STYLES = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }

    .hero {
        text-align: center;
        padding: 2rem 1rem 1.5rem;
        margin-bottom: 1rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #e94560, #ff6b9d, #c77dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #a0aec0;
        font-size: 1.05rem;
        margin-top: 0.6rem;
    }

    .recorder-box {
        background: rgba(255,255,255,0.03);
        border: 1px dashed rgba(233, 69, 96, 0.4);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
    }

    .recorder-label {
        color: #cbd5e0;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }

    .chat-user {
        background: linear-gradient(135deg, #2d3748, #1a202c);
        border-left: 4px solid #e94560;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
    }

    .chat-assistant {
        background: linear-gradient(135deg, #1e3a5f, #162447);
        border-left: 4px solid #48bb78;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
    }

    .chat-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
    }

    .chat-user .chat-label { color: #fc8181; }
    .chat-assistant .chat-label { color: #68d391; }

    .chat-text {
        color: #e2e8f0;
        font-size: 1rem;
        line-height: 1.6;
    }

    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #718096;
    }

    .empty-state .icon { font-size: 3rem; margin-bottom: 0.5rem; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }

    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 500;
        background: rgba(72, 187, 120, 0.15);
        color: #68d391;
        border: 1px solid rgba(72, 187, 120, 0.3);
    }
</style>
"""


def inject_styles():
    # FIX: Streamlit reruns the whole script on every interaction. Injecting
    # ~100 lines of <style> markdown every single rerun adds real, avoidable
    # overhead. Inject once per session instead.
    if not st.session_state.get("_styles_injected"):
        st.markdown(STYLES, unsafe_allow_html=True)
        st.session_state["_styles_injected"] = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def init_session():
    defaults = {
        "messages": [],
        "history": [],
        "total_tokens": 0,
        "last_audio_hash": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> genai.Client:
    # FIX: st.cache_resource is the correct primitive for "expensive object
    # that should survive reruns and be shared/reused" — it replaces the
    # manual session_state bookkeeping and guarantees the client (and its
    # underlying HTTP connection pool) isn't rebuilt on every rerun.
    return genai.Client(api_key=api_key)


def add_tokens(count: int):
    st.session_state.total_tokens += count


def render_message(role: str, text: str, audio_bytes: bytes | None = None):
    label = "You" if role == "user" else "AudioFlow"
    css_class = "chat-user" if role == "user" else "chat-assistant"
    safe_text = html.escape(text).replace("\n", "<br>")

    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="chat-label">{label}</div>
            <div class="chat-text">{safe_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")


def render_chat():
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-state">
                <div class="icon">🎙️</div>
                <p>No conversation yet.<br>Press the mic below and start talking!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for msg in st.session_state.messages:
        render_message(msg["role"], msg["text"], msg.get("audio"))


def process_recording(client: genai.Client, recorded_audio: bytes):
    # FIX: cheaper, faster hash than md5 for a simple "did this change" check.
    audio_hash = hashlib.sha1(recorded_audio).hexdigest()
    if st.session_state.last_audio_hash == audio_hash:
        return
    st.session_state.last_audio_hash = audio_hash

    # FIX: avoid tempfile + disk writes/reads. Every recording previously
    # did: write input.wav to disk -> read it back for transcription,
    # then write response.wav to disk -> read it back for playback. That's
    # 4 extra disk I/O ops per turn for no benefit, since the underlying
    # SDK calls can take in-memory bytes just as easily. If your
    # transcribe_audio/text_to_audio signatures currently require a file
    # PATH rather than bytes, update them to accept file-like objects
    # (io.BytesIO) or raw bytes instead — that's the other half of this fix.
    audio_buf = io.BytesIO(recorded_audio)
    audio_buf.name = "input.wav"  # some SDKs use .name to infer mime type

    with st.status("Processing your voice...", expanded=True) as status:
        st.write("Transcribing speech...")
        try:
            transcribed, tokens = transcribe_audio(client, audio_buf)
            add_tokens(tokens)
        except Exception as exc:
            status.update(label="Transcription failed", state="error")
            st.error(f"Could not understand audio: {exc}")
            return

        if not transcribed:
            status.update(label="No speech detected", state="error")
            st.warning("Didn't catch any speech — try again.")
            return

        st.write("Generating reply...")
        try:
            reply, tokens = fetch_ai_response(client, transcribed, st.session_state.history)
            add_tokens(tokens)
        except Exception as exc:
            status.update(label="AI response failed", state="error")
            st.error(f"AI error: {exc}")
            return

        st.write("Creating voice...")
        audio_bytes = None
        try:
            response_buf = io.BytesIO()
            text_to_audio(client, reply, response_buf)
            audio_bytes = response_buf.getvalue()
        except Exception as exc:
            st.warning(f"Text reply ready, but voice failed: {exc}")

        status.update(label="Done!", state="complete")

    st.session_state.history.append({"user": transcribed, "assistant": reply})
    st.session_state.messages.append({"role": "user", "text": transcribed})
    st.session_state.messages.append({"role": "assistant", "text": reply, "audio": audio_bytes})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    init_session()
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>🎙️ AudioFlow</h1>
            <p>Speak naturally — I'll listen, think, and talk back.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    env_key = get_api_key()

    with st.sidebar:
        st.header("Settings")

        if env_key:
            st.success("API key loaded from .env")
            api_key = env_key
        else:
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="Paste your key here",
                help="Get a free key at aistudio.google.com/apikey",
            )

        st.divider()

        st.metric("Tokens used", st.session_state.total_tokens)
        st.caption(f"Model: `{TEXT_MODEL}`")
        st.caption(f"TTS: `{TTS_MODEL}` · voice `{TTS_VOICE}`")
        st.caption(f"Max output: {MAX_OUTPUT_TOKENS} · thinking: {THINKING_LEVEL}")

        st.divider()

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history = []
            st.session_state.last_audio_hash = None
            st.rerun()

        st.markdown(
            '<span class="status-pill">● Ready</span>',
            unsafe_allow_html=True,
        )

    if not api_key:
        st.info("Add your Gemini API key in the sidebar, or create a `.env` file (see `.env.example`).")
        st.link_button("Get a free API key →", "https://aistudio.google.com/apikey")
        return

    client = get_client(api_key)

    st.subheader("Conversation")
    conversation_area = st.empty()
    with conversation_area.container():
        render_chat()

    st.markdown('<div class="recorder-box">', unsafe_allow_html=True)
    st.markdown('<p class="recorder-label">Tap to record · tap again to stop</p>', unsafe_allow_html=True)

    recorded = audio_recorder(
        text="",
        recording_color="#e94560",
        neutral_color="#4a5568",
        icon_name="microphone",
        icon_size="3x",
        key="voice_recorder",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if recorded:
        process_recording(client, recorded)
        conversation_area.empty()
        with conversation_area.container():
            render_chat()


if __name__ == "__main__":
    main()