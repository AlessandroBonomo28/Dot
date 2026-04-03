import pyaudio
import numpy as np
import threading
import queue
import re
from openvino_genai import WhisperPipeline

pipe = WhisperPipeline("./whisper-small-ov", device="NPU")

RATE = 16000
CHUNK = 1024
CHANNELS = 1

RECORD_SECONDS = 5  
frames_per_period = int(RATE / CHUNK * RECORD_SECONDS)

coda_audio = queue.Queue()

# Ora riceve direttamente l'array numpy già elaborato
def trascrivi(audio_np):
    result = pipe.generate(audio_np)
    return str(result).strip()



# --- THREAD: Gestisce la coda, calcola il volume e trascrive ---
def elaboratore_trascrizioni():
    while True:
        frames_da_elaborare = coda_audio.get()
        
        if frames_da_elaborare is None:
            break
            
        # 1. Uniamo i frame e li convertiamo in formato numerico (float32)
        audio_bytes = b''.join(frames_da_elaborare)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 2. CALCOLO DEL VOLUME MEDIO (RMS) DEL BLOCCO INTERO
        volume_medio = np.sqrt(np.mean(audio_np**2))
        
        # Stampa di debug in giallo (se il terminale lo supporta) o normale
        print(f"   [DEBUG VOLUME] -> {volume_medio:.5f}")
        
        # 3. Inviamo alla NPU
        if (volume_medio < 0.00011):
            print("skip")
            continue
        testo = trascrivi(audio_np)

        if re.search(r'(.{3,}?)(,?\s+\1){4,}', testo, re.IGNORECASE):
            print("non valido")
            coda_audio.task_done()
            continue
        if len(testo)==1:
            print("<=1")
            coda_audio.task_done()
            continue
        

        # Pattern for Chinese (CJK) and Ukrainian (Cyrillic)
        # Includes CJK Unified Ideographs, CJK Extensions, and Cyrillic
        pattern = re.compile(r'[\u0400-\u04ff\u4e00-\u9fff\u3400-\u4dbf]+')
        if bool(pattern.search(testo)):
            print("circillico")

        if re.search(r'(.)\1{9,}', testo):
            print("pattern ................")
            coda_audio.task_done()
            continue
        
        if testo: 
            print(f"[NPU] >> {testo}")
            
        coda_audio.task_done()

thread_trascrizione = threading.Thread(target=elaboratore_trascrizioni, daemon=True)
thread_trascrizione.start()

# --- LOOP PRINCIPALE: Ascolta e taglia a fette ---
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

print(f"Trascrizione continua avviata. Registrazione a blocchi di {RECORD_SECONDS}s.")
print("Fai silenzio per vedere il livello di rumore di fondo. Ctrl+C per fermare\n")

frames = []

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

        if len(frames) >= frames_per_period:
            coda_audio.put(frames)
            frames = [] 

except KeyboardInterrupt:
    print("\nFermato. Attendo fine elaborazione coda...")
    coda_audio.put(None)
    thread_trascrizione.join()
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Chiusura completata.")