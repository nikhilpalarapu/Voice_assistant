import os
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel

import voice_service as vs
from rag.AIVoiceAssistant import AIVoiceAssistant

# ---------- Tunables ----------
DEFAULT_MODEL_SIZE = "base"          # a bit more accurate than "small"; use "base.en" for only-English
DEFAULT_CHUNK_LENGTH = 3             # seconds
MIC_INDEX = None                     # set to an int after you list devices (optional)
WELCOME_MSG = "Welcome to Bangalore Kitchen. What is your name?"
GOODBYE_MARKERS = ("goodbye", "thanks for visiting")  # phrases that mean "end the session"
# --------------------------------

ai_assistant = AIVoiceAssistant()

# --- list input devices once (optional) ---
if __name__ == "__main__" and False:  # set to True, run once, read indices, set MIC_INDEX, then back to False
    pa = pyaudio.PyAudio()
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        print(i, info.get('name'), "in_channels=", info.get('maxInputChannels'))
    pa.terminate()
    raise SystemExit


def is_silence(int16_samples, mean_abs_thresh=20.0, peak_thresh=350):
    x = int16_samples.astype(np.float32)
    if x.ndim > 1:
        x = x[:, 0]
    mean_abs = np.mean(np.abs(x))
    rms = np.sqrt(np.mean(x**2))
    peak = np.max(np.abs(x))
    print(f"[audio] mean_abs={mean_abs:.1f} rms={rms:.1f} peak={peak:.0f}")
    # Not silent if either mean above floor OR clear peak present
    return not (mean_abs >= mean_abs_thresh or peak >= peak_thresh)


def record_audio_chunk(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)

    temp_file_path = "temp_audio_chunk.wav"
    with wave.open(temp_file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b"".join(frames))

    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data, mean_abs_thresh=20.0, peak_thresh=350):
            os.remove(temp_file_path)
            return True   # still silent
        else:
            return False  # voice detected
    except Exception as e:
        print(f"Error while reading audio file: {e}")
        return False


def transcribe_audio(model, file_path):
    # You can add vad_filter=True if your faster-whisper version supports it:
    # segments, info = model.transcribe(file_path, beam_size=5, vad_filter=True)
    segments, info = model.transcribe(file_path, beam_size=5)
    transcription = " ".join(segment.text for segment in segments)
    return transcription.strip()


def should_end_session(assistant_text: str) -> bool:
    t = (assistant_text or "").lower()
    return any(key in t for key in GOODBYE_MARKERS)


def maybe_handle_voice_command(user_text: str) -> bool:
    """
    Returns True if a special command was handled (and we should skip LLM turn).
    Supported:
      - "show orders" / "orders" / "order status" / "view orders"
      - "reset" / "new order" / "start over"
      - "clear orders" (clears in-memory list; CSV remains)
    """
    cmd = (user_text or "").lower().strip()
    if not cmd:
        return False

    # View orders
    if cmd in ("show orders", "orders", "order status", "view orders"):
        orders = ai_assistant.get_orders()
        if not orders:
            msg = "No orders yet."
        else:
            lines = [f"{o['timestamp']}: {o['name']} -> {o['item']} (₹{o['price']})" for o in orders]
            msg = "Here are your orders:\n" + "\n".join(lines)
        print(msg)
        vs.play_text_to_speech("Showing orders in the console.")
        return True

    # Reset session
    if cmd in ("reset", "new order", "start over"):
        ai_assistant.reset_session()
        vs.play_text_to_speech("Session reset. What is your name?")
        print("AI Assistant: Session reset. What is your name?")
        return True

    # Clear only in-memory orders
    if cmd in ("clear orders", "clear in memory orders", "clear session orders"):
        ai_assistant.clear_orders()
        vs.play_text_to_speech("In-memory orders cleared.")
        print("AI Assistant: In-memory orders cleared.")
        return True

    return False


def main():
    # Whisper
    model_size = DEFAULT_MODEL_SIZE
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # Mic
    audio = pyaudio.PyAudio()
    stream_kwargs = dict(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    if MIC_INDEX is not None:
        stream_kwargs["input_device_index"] = MIC_INDEX
    stream = audio.open(**stream_kwargs)

    # 1) Speak the welcome message immediately (no need to say "hi")
    vs.play_text_to_speech(WELCOME_MSG)
    print("AI Assistant:", WELCOME_MSG)

    try:
        while True:
            print("_")  # recording tick
            if not record_audio_chunk(audio, stream):
                chunk_file = "temp_audio_chunk.wav"
                user_text = transcribe_audio(model, chunk_file)
                os.remove(chunk_file)

                if not user_text:
                    continue

                print("Customer:", user_text)

                # 2) Special voice commands (show orders / reset / clear)
                if maybe_handle_voice_command(user_text):
                    continue  # handled – go listen again

                # 3) Drive assistant + TTS
                reply = ai_assistant.interact_with_llm(user_text)
                if reply:
                    reply = reply.strip()
                    vs.play_text_to_speech(reply)
                    print("AI Assistant:", reply)

                    # 4) Clean exit after "Goodbye"
                    if should_end_session(reply):
                        break

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    main()
