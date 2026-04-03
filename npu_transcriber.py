import librosa
from openvino_genai import WhisperPipeline

# 1. Carica il modello sulla NPU
pipe = WhisperPipeline("./whisper-small-ov", device="NPU")

# 2. Carica il file audio
# sr=16000 forza il campionamento richiesto da Whisper
file_audio = "test.wav" 
audio, _ = librosa.load(file_audio, sr=16000)

# 3. Trascrivi
result = pipe.generate(audio)

print(f"Risultato: {result}")