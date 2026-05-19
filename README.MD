# 🎙️  Hologram AI & Myanmar Voice Assistant

A real-time AI voice chat application powered by **Google Gemini Live API**, configured to speak and respond exclusively in the **Myanmar (Burmese) language**.

---

## 🎥 Demo

![Demo](./img/demo.gif)

> 🎬 Or watch the full demo on YouTube: [https://youtube.com/your-link-here](https://youtube.com/your-link-here)

---

## 🖼️ Preview

![Voice Chat Preview](./img/demo.png)

---

## ✨ Features

- 🎤 Real-time microphone input streaming
- 🔊 Live audio response playback
- 🇲🇲 Myanmar language only mode (`my-MM`)
- 🤖 Powered by Gemini Live (`gemini-3.1-flash-live-preview`)

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core language |
| **Google Gemini Live API** | Real-time AI voice model |
| **PyAudio** | Microphone input & speaker output |
| **AsyncIO** | Concurrent audio streaming |
| **Python-dotenv** | Secure API key management |

---

## 🔄 Architecture Flow

```
Microphone 🎤
    ↓  (PCM Audio @ 16000 Hz)
Gemini Live API 🤖
    ↓  (AI Response in Myanmar)
Speaker 🔊  (PCM Audio @ 24000 Hz)
```

---

## 📋 Requirements

- Python 3.9+
- A valid **Google Gemini API Key**
- A working microphone and speaker

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Mac/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install google-genai pyaudio python-dotenv
   ```

4. **Set up your `.env` file**
   ```bash
   cp .env.example .env
   ```
   Then open `.env` and replace with your actual Gemini API key:
   ```
   API_KEY=your_gemini_api_key_here
   ```

---

## 🚀 Usage

```bash
python ai_voice_chat.py
```

After launching the application, you should see:
```
Connecting to gemini-3.1-flash-live-preview in Myanmar Mode...
မင်္ဂလာပါ။ အခု မြန်မာလို ပြောလို့ရပါပြီ။
```

Just start speaking — Gemini will respond in the Myanmar (Burmese) language through your speaker.

---

## ⚙️ Configuration

| Setting | Value | Description |
|---|---|---|
| `MODEL` | `gemini-3.1-flash-live-preview` | Gemini Live model |
| `INPUT_RATE` | `16000 Hz` | Microphone sample rate |
| `OUTPUT_RATE` | `24000 Hz` | Speaker sample rate |
| `CHANNELS` | `1` | Mono audio |
| `language_code` | `my-MM` | Myanmar language |
| `voice_name` | `Puck` | Gemini prebuilt voice |

---

## 📁 Project Structure

```
.
├── ai_voice_chat.py   # Main application
├── .env               # API key (never commit this)
├── .gitignore         # Git ignore rules
├── img/
│   ├── demo.gif       # Demo animation
│   └── demo.png       # Preview screenshot
└── README.md          # This file
```

---

## 🔑 Getting a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click **Get API Key** and create a new key
4. Paste it into your `.env` file

---

## ⚠️ Notes

- Never commit your `.env` file — it's already in `.gitignore`
- `pyaudio` may require additional system dependencies on some platforms:
  - **Mac:** `brew install portaudio`
  - **Linux:** `sudo apt-get install portaudio19-dev`
  - **Windows:** install via `pipwin install pyaudio`

---

## 🚧 Future Improvements

- [ ] Add emotion-aware voice responses
- [ ] Add real-time streaming subtitles
- [ ] Add GUI interface
- [ ] Support multiple Myanmar dialects
- [ ] Add wake-word detection

---

## 📄 License

MIT License — feel free to use and modify.