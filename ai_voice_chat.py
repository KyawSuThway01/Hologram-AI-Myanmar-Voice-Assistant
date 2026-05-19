import asyncio
import pyaudio
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv


load_dotenv()

# --- Config ---
API_KEY = os.getenv("API_KEY")
MODEL = "gemini-3.1-flash-live-preview"

# Audio Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK_SIZE = 512


async def main():
    client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1alpha'})
    audio = pyaudio.PyAudio()

    mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=INPUT_RATE, input=True, frames_per_buffer=CHUNK_SIZE)
    speaker_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=OUTPUT_RATE, output=True,
                                frames_per_buffer=CHUNK_SIZE)

    # UPDATED CONFIG FOR MYANMAR LANGUAGE
    config = {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}},
            "language_code": "my-MM"  # Sets the primary language to Myanmar
        },
        # Strong instructions to keep the robot speaking only Myanmar
        "system_instruction": "You are a helpful robot friend. YOU MUST SPEAK ONLY IN MYANMAR LANGUAGE. Do not use English even if I speak English. Always reply in clear Burmese/Myanmar."
    }

    print(f"Connecting to {MODEL} in Myanmar Mode...")

    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            print("မင်္ဂလာပါ။ အခု မြန်မာလို ပြောလို့ရပါပြီ။ (Ready! Speak in Myanmar now.)")

            async def send_audio():
                try:
                    while True:
                        data = mic_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                        await session.send_realtime_input(
                            audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                        )
                        await asyncio.sleep(0.005)
                except Exception as e:
                    print(f"Send error: {e}")

            async def receive_audio():
                try:
                    while True:
                        async for response in session.receive():
                            if response.server_content:
                                content = response.server_content
                                if content.model_turn:
                                    for part in content.model_turn.parts:
                                        if part.inline_data:
                                            speaker_stream.write(part.inline_data.data)

                                if content.turn_complete:
                                    print("\n[Gemini finished] - နားထောင်နေပါတယ်။")
                        await asyncio.sleep(0.1)
                except Exception as e:
                    print(f"Receive error: {e}")

            await asyncio.gather(send_audio(), receive_audio())

    except Exception as e:
        print(f"Connection closed: {e}")
    finally:
        mic_stream.close()
        speaker_stream.close()
        audio.terminate()


if __name__ == "__main__":
    asyncio.run(main())
