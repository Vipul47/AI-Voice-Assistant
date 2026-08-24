import base64
import time
import wave

from google import genai

from config import MAX_OUTPUT_TOKENS, TEXT_MODEL, THINKING_LEVEL, TTS_MODEL, TTS_VOICE


def _token_count(response) -> int:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return 0
    return getattr(usage, "total_token_count", 0) or 0


def _generation_config() -> dict:
    """Return generation settings in the Interactions API schema."""
    return {
        "thinking_level": THINKING_LEVEL,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }


def transcribe_audio(client: genai.Client, audio_path, mime_type: str = "audio/wav") -> tuple[str, int]:
    # FIX: files.upload() can only infer the mime type from a real file
    # path's extension (e.g. ".wav"). When called with an in-memory buffer
    # (io.BytesIO) instead of a path, there's no extension to infer from,
    # which is what caused "Unknown mime type: Could not determine the
    # mimetype for your file". Passing mime_type explicitly fixes it
    # whether audio_path is a real path or an in-memory buffer.
    audio_file = client.files.upload(file=audio_path, config={"mime_type": mime_type})
    try:
        response = client.interactions.create(
            model=TEXT_MODEL,
            input=[
                {
                    "type": "text",
                    "text": "Transcribe this audio exactly. Return only the spoken text, nothing else.",
                },
                {
                    "type": "audio",
                    "uri": audio_file.uri,
                    "mime_type": audio_file.mime_type,
                },
            ],
            generation_config=_generation_config(),
        )
    finally:
        try:
            client.files.delete(name=audio_file.name)
        except Exception:
            pass

    return (response.output_text or "").strip(), _token_count(response)


def fetch_ai_response(
    client: genai.Client,
    input_text: str,
    history: list[dict] | None = None,
) -> tuple[str, int]:
    history_lines = []
    for turn in history or []:
        history_lines.append(f"User: {turn['user']}\nAssistant: {turn['assistant']}")

    prompt_parts = []
    if history_lines:
        prompt_parts.append("Previous conversation:\n" + "\n\n".join(history_lines))
    prompt_parts.append(f"User: {input_text}")

    response = client.interactions.create(
        model=TEXT_MODEL,
        input=(
            "You are a friendly voice assistant. Answer clearly and concisely "
            "in a few sentences—your reply will be read aloud.\n\n"
            + "\n\n".join(prompt_parts)
        ),
        generation_config=_generation_config(),
    )
    return (response.output_text or "").strip(), _token_count(response)


def text_to_audio(client: genai.Client, text: str, audio_path, attempts: int = 2) -> bool:
    last_error = None

    for _ in range(attempts):
        try:
            stream = client.interactions.create(
                model=TTS_MODEL,
                input=text,
                response_format={"type": "audio"},
                generation_config={"speech_config": [{"voice": TTS_VOICE}]},
                stream=True,
            )

            audio_chunks = []
            for event in stream:
                if event.event_type == "step.delta" and event.delta.type == "audio":
                    audio_chunks.append(base64.b64decode(event.delta.data))

            audio_data = b"".join(audio_chunks)
            if not audio_data:
                raise RuntimeError("No audio returned by the TTS model.")

            # NOTE: wave.open() accepts either a path string or a
            # file-like object that supports write/seek (io.BytesIO
            # qualifies), so this works unchanged for both call styles.
            with wave.open(audio_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio_data)

            return True

        except Exception as exc:
            last_error = exc
            time.sleep(1)

    raise RuntimeError(f"Voice generation failed: {last_error}")