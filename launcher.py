#!/usr/bin/env python3
"""
launcher.py  —  Voice-controlled GLB model switcher (Burmese / Myanmar)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Place this file in the SAME folder as app.py and ai_voice_chat.py.
Put your GLB files in the same folder (filenames = model names).

  Example:  Anawrahta.glb   →  AI says "အနော်ရထာ မင်းတရားကို ပြသနေပါသည်"
            Bayinnaung.glb  →  AI says "ဘုရင့်နောင် မင်းတရား ..."

Run:
    python launcher.py

Say 'ပြောင်း' / 'နောက်' / 'တစ်ခြား' to switch models.
The AI will announce the new model's name in Burmese automatically.

Neither app.py nor ai_voice_chat.py is modified.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import os
import threading
import asyncio
import queue
import glob
import math


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Background Music — put your audio file next to launcher.py and set the
#  filename below.  Supports .wav, .mp3, .ogg (anything pyglet can load).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MUSIC_FILE          = "background.wav"   # ← change to your filename
MUSIC_VOLUME_NORMAL = 0.15               # normal volume  (0.0 – 1.0)
MUSIC_VOLUME_DUCK   = 0.04              # volume while AI is speaking
MUSIC_FADE_SPEED    = 0.8               # fade speed in units/second

_ai_speaking = False   # set True/False by the voice thread


class BackgroundMusic:

    def __init__(self):
        self._player = None
        self._volume = MUSIC_VOLUME_NORMAL

    def start(self):
        path = os.path.join(HERE, MUSIC_FILE)
        if not os.path.isfile(path):
            print(f"[Music] File not found: {path}  — no background music.")
            return
        try:
            source = pyglet.media.load(path, streaming=True)
            self._player = pyglet.media.Player()
            self._player.loop = True
            self._player.volume = MUSIC_VOLUME_NORMAL
            self._player.queue(source)
            self._player.play()
            print(f"[Music] Playing '{MUSIC_FILE}' at {MUSIC_VOLUME_NORMAL:.0%} volume")
        except Exception as e:
            print(f"[Music] Could not start: {e}")
            self._player = None

    def tick(self, dt):
        if self._player is None:
            return
        target = MUSIC_VOLUME_DUCK if _ai_speaking else MUSIC_VOLUME_NORMAL
        diff = target - self._volume
        step = MUSIC_FADE_SPEED * dt
        self._volume = target if abs(diff) <= step else self._volume + math.copysign(step, diff)
        self._player.volume = max(0.0, min(1.0, self._volume))

    def stop(self):
        if self._player:
            self._player.pause()
            self._player.delete()
            self._player = None


_bg_music = BackgroundMusic()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Auto-detect GLB files  (or hard-code MODEL_PATHS below)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _auto_find_glbs():
    files = (
        glob.glob(os.path.join(HERE, "*.glb"))   +
        glob.glob(os.path.join(HERE, "*.gltf"))  +
        glob.glob(os.path.join(HERE, "models", "*.glb"))  +
        glob.glob(os.path.join(HERE, "models", "*.gltf"))
    )
    return sorted(set(files))

MODEL_PATHS = _auto_find_glbs()
# MODEL_PATHS = ["Anawrahta.glb", "Bayinnaung.glb", "Tabinshwehti.glb"]  # hard-code option

if not MODEL_PATHS:
    print("ERROR: No GLB/GLTF files found. Put your models next to launcher.py.")
    sys.exit(1)

print(f"[Launcher] {len(MODEL_PATHS)} model(s) found:")
for i, p in enumerate(MODEL_PATHS):
    print(f"  [{i+1}] {os.path.basename(p)}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Shared queues between voice thread <-> viewer
#
#  model_switch_queue  : voice thread puts (glb_path, model_name) → viewer loads it
#  announce_queue      : viewer confirms load → voice thread sends text to Gemini
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
model_switch_queue = queue.Queue()   # (path, name)  viewer-side
announce_queue     = queue.Queue()   # name string   voice-side

_model_index = [0]

def _get_model_name(path: str) -> str:
    """Strip folder + extension to get the human name, e.g. 'Anawrahta'."""
    return os.path.splitext(os.path.basename(path))[0]

def _next_model() -> tuple:
    _model_index[0] = (_model_index[0] + 1) % len(MODEL_PATHS)
    path = MODEL_PATHS[_model_index[0]]
    return path, _get_model_name(path)

# Burmese (and fallback English) trigger words for "change model"
SWITCH_KEYWORDS = [
    "နောက်", "ပြောင်း", "တစ်ခြား", "အခြား", "ကြည့်", "ပြ",
    "မော်ဒယ်", "နောက်တစ်ခု", "ပြောင်းပေး", "ကြည့်ချင်",
    "next", "change", "model", "show",
]

def _detected_switch(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in SWITCH_KEYWORDS)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Voice thread
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import ai_voice_chat
from google import genai
from google.genai import types
import pyaudio

async def _voice_main():
    FORMAT      = ai_voice_chat.FORMAT
    CHANNELS    = ai_voice_chat.CHANNELS
    INPUT_RATE  = ai_voice_chat.INPUT_RATE
    OUTPUT_RATE = ai_voice_chat.OUTPUT_RATE
    CHUNK_SIZE  = ai_voice_chat.CHUNK_SIZE
    API_KEY     = ai_voice_chat.API_KEY
    MODEL_ID    = ai_voice_chat.MODEL

    client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1alpha'})
    audio  = pyaudio.PyAudio()

    mic_stream = audio.open(
        format=FORMAT, channels=CHANNELS,
        rate=INPUT_RATE, input=True, frames_per_buffer=CHUNK_SIZE
    )
    speaker_stream = audio.open(
        format=FORMAT, channels=CHANNELS,
        rate=OUTPUT_RATE, output=True, frames_per_buffer=CHUNK_SIZE
    )

    # First model name for the opening greeting
    first_name = _get_model_name(MODEL_PATHS[0])

    config = {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}},
            "language_code": "my-MM",
        },
        "system_instruction": (
            "သင်သည် မြန်မာဘာသာဖြင့်သာ ပြောဆိုသည့် ရင်းနှီးသော ကူညီသည့် AI ဖြစ်သည်။ "
            "အသုံးပြုသူက မြန်မာပြောလျှင် မြန်မာလိုပြောရမည်။ "
            "English လိုပြောလျှင် English လိုတာပြန်ပြောရမည်။ မြန်မာ နဲ့ English နှစ်ဘာသာပဲသုံးရမည်။ အသုံးပြုသူဘမော်ဒယ်ပြောင်းရန်တောင်းဆိုသောအခါ မော်ဒင်ပြောင်းနေကြောင်းပြောရမည်။ "
            "မေးခွန်းတစ်ခု‌ ‌ဖြေပြီးတိုင်  မြန်မာတွေ တော်ကြောင်း ဩဇာကြီးခဲ့ကြောင့် အားပေးစရာတွေ ထည့်ပြောရမည် "
        ),
        "input_audio_transcription":  {},
        "output_audio_transcription": {},
    }

    print(f"[Voice] Connecting to {MODEL_ID} ...")

    try:
        async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
            print("[Voice] မင်္ဂလာပါ။ မြန်မာလို ပြောလို့ရပြီ။")
            print(f"[Voice] First model: {first_name}")

            # Greet with the first model name right away
            await session.send_client_content(
                turns=[{
                    "role": "user",
                    "parts": [{"text":
                        f"စတင်နေပါပြီ။ ယခု ပထမဆုံး မော်ဒယ် '{first_name}' ကို ပြသနေပါသည်။ "
                        f"မြန်မာဘာသာဖြင့် ကြိုဆိုနှုတ်ဆက်ပြီး '{first_name}' ဘယ်သူ/ဘာဆိုတာ တိုတောင်းစွာ မိတ်ဆက်ပေးပါ။"
                    }]
                }],
                turn_complete=True
            )

            async def send_audio():
                try:
                    while True:
                        data = mic_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                        await session.send_realtime_input(
                            audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                        )
                        await asyncio.sleep(0.005)
                except Exception as e:
                    print(f"[Voice] Send error: {e}")

            async def watch_announce():
                """
                Polls announce_queue (thread-safe) and sends a text prompt
                into the live session so Gemini announces the new model in Burmese.
                """
                try:
                    while True:
                        try:
                            name = announce_queue.get_nowait()
                            prompt = (
                                f"မော်ဒယ် ပြောင်းပြီးပါပြီ။ "
                                f"ယခု '{name}' ကို ပြသနေပါသည်။ "
                                f"မြန်မာဘာသာဖြင့် '{name}' ဘယ်သူ/ဘာဆိုတာ တိုတောင်းစွာ မိတ်ဆက်ပေးပါ။"
                            )
                            print(f"[Voice] Sending announcement for: {name}")
                            await session.send_client_content(
                                turns=[{"role": "user", "parts": [{"text": prompt}]}],
                                turn_complete=True
                            )
                        except queue.Empty:
                            pass
                        await asyncio.sleep(0.1)
                except Exception as e:
                    print(f"[Voice] Announce error: {e}")

            async def receive_audio():
                global _ai_speaking
                try:
                    while True:
                        async for response in session.receive():
                            if response.server_content:
                                content = response.server_content

                                # Play AI audio — duck the background music
                                if content.model_turn:
                                    for part in content.model_turn.parts:
                                        if part.inline_data:
                                            _ai_speaking = True   # duck music
                                            speaker_stream.write(part.inline_data.data)

                                # Check user speech for switch command
                                if (hasattr(content, 'input_transcription')
                                        and content.input_transcription):
                                    text = content.input_transcription.text or ""
                                    if text:
                                        print(f"[Voice] Heard: {text}")
                                        if _detected_switch(text):
                                            next_path, next_name = _next_model()
                                            print(f"[Voice] ➜ Queuing switch to: {next_name}")
                                            # Tell viewer to load the new GLB
                                            model_switch_queue.put((next_path, next_name))

                                if content.turn_complete:
                                    _ai_speaking = False  # restore music volume
                                    print("[Voice] နားထောင်နေပါတယ်။")

                        await asyncio.sleep(0.1)
                except Exception as e:
                    print(f"[Voice] Receive error: {e}")

            # Run all three coroutines concurrently
            await asyncio.gather(send_audio(), receive_audio(), watch_announce())

    except Exception as e:
        print(f"[Voice] Connection closed: {e}")
    finally:
        mic_stream.close()
        speaker_stream.close()
        audio.terminate()


def _run_voice():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_voice_main())
    except Exception as e:
        print(f"[Voice] Thread error: {e}")
    finally:
        loop.close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  VoiceModelViewer — subclass of app.ModelViewer (app.py untouched)
#  Every tick: checks model_switch_queue, hot-swaps GLB, puts name in announce_queue
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import app as _app
import pyglet  # already imported at top for BackgroundMusic, re-affirmed here

class VoiceModelViewer(_app.ModelViewer):

    def _tick(self, dt):
        super()._tick(dt)   # parent: gestures, zoom lerp, rotation lerp

        # Smoothly duck/unduck background music each frame
        _bg_music.tick(dt)

        try:
            new_path, new_name = model_switch_queue.get_nowait()
        except queue.Empty:
            return

        print(f"[Viewer] Loading: {new_name}  ({os.path.basename(new_path)})")
        for m in self.meshes:
            m.delete()

        try:
            self.meshes, self.base_zoom = _app.load_glb_scene(new_path)
            PHI_DEFAULT             = math.pi / 3
            self.camera_theta       = 0.0
            self.camera_phi         = PHI_DEFAULT
            self.target_theta       = 0.0
            self.target_phi         = PHI_DEFAULT
            self.current_zoom_scale = 1.0
            self.target_zoom_scale  = 1.0
            self.set_caption(f"GLB Viewer — {new_name}")
            print(f"[Viewer] ✓ Switched to: {new_name}")

            # Signal voice thread to make Gemini announce this model
            announce_queue.put(new_name)

        except Exception as e:
            print(f"[Viewer] Failed to load {new_path}: {e}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    voice_thread = threading.Thread(target=_run_voice, daemon=True, name="VoiceChat")
    voice_thread.start()
    print("[Launcher] Voice thread started.")

    first_model = MODEL_PATHS[0]
    print(f"[Launcher] Opening: {_get_model_name(first_model)}")
    viewer = VoiceModelViewer(first_model)

    # Start background music now that pyglet's audio system is up
    _bg_music.start()

    try:
        pyglet.app.run()
    finally:
        _bg_music.stop()