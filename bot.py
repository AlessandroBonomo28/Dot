import time
import json
import telepot
import os,re
import librosa
import numpy as np
from telepot.loop import MessageLoop
from openvino_genai import WhisperPipeline
from dotenv import load_dotenv
from client_llm import LLMClient
# --- CONFIGURAZIONE INIZIALE ---
load_dotenv()

TOKEN = os.getenv('TOKEN')
if not TOKEN:
    print("ERROR: Token non trovato nel file .env!")
    exit(1)


ENFORCE_WHITELIST = True
whitelist_file = "whitelist.json"

whitelist = []

if not os.path.exists(whitelist_file):
    with open(whitelist_file, 'w') as f:
        json.dump([], f)
    print(f"Created {whitelist_file} with empty array")
else:
    print(f"{whitelist_file} already exists")

with open(whitelist_file, "r") as f:
    whitelist = json.load(f)

bot = telepot.Bot(TOKEN)
pipe = WhisperPipeline("./whisper-small-ov", device="NPU")
llm = LLMClient(provider="ollama", model="qwen3-vl:8b")

chat_histories = {}
MAX_HISTORY = 10

def add_to_history(chat_id, role, content):
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
    chat_histories[chat_id].append({"role": role, "content": content})
    # Mantieni solo gli ultimi N messaggi
    if len(chat_histories[chat_id]) > MAX_HISTORY:
        chat_histories[chat_id] = chat_histories[chat_id][-MAX_HISTORY:]

def trascrivi_audio(file_path):
    """Carica il file, lo ricampiona e lo trascrive."""
    try:
        # Carica e forza a 16kHz (richiesto da Whisper)
        audio, _ = librosa.load(file_path, sr=16000)
        # Genera trascrizione
        result = pipe.generate(audio)
        testo = str(result).strip()
        if re.search(r'(.{3,}?)(,?\s+\1){4,}', testo, re.IGNORECASE):
            raise 'errore 1'
        if len(testo)==1:
            raise 'errore 2'
        if re.search(r'(.)\1{9,}', testo):
            raise 'errore 3'
        return testo
    except Exception as e:
        return f"Errore durante la trascrizione: {e}"

# --- GESTORE MESSAGGI TELEGRAM ---
def handle(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    chat_id = str(chat_id)
    text = msg.get('text', '').strip()

    if chat_id not in whitelist and ENFORCE_WHITELIST:
        bot.sendMessage(chat_id, f"{chat_id} non autorizzato.")
        return

    # Gestione Testo (Comandi)
    if content_type == 'text':
        text = msg.get('text', '').strip()
        if text == '/start':
            bot.sendMessage(chat_id, "Hello! Inviami un messaggio vocale o un file audio e lo trascriverò per te.")
            return
        else:
            bot.sendMessage(chat_id, "Pensando a una risposta...")
            risposta = llm.ask(text,past_messages=chat_histories.get(chat_id, []))
            add_to_history(chat_id, "user", text)
            add_to_history(chat_id, "assistant", risposta)
            bot.sendMessage(chat_id, risposta)
            
            
    if content_type == 'photo':
        bot.sendMessage(chat_id, "Sto guardando l'immagine...")
        
        # Prendi la versione più grande della foto
        file_id = msg['photo'][-1]['file_id']
        temp_img = f"img_{chat_id}.jpg"
        
        bot.download_file(file_id, temp_img)
        
        # Chiedi a Qwen di descrivere l'immagine
        bot.sendMessage(chat_id, "Pensando a una risposta...")
        didascalia = msg.get('caption', 'Cosa vedi in questa immagine?')
        print("didascalia:",didascalia)
        risposta = llm.ask_vision(didascalia, temp_img,past_messages=chat_histories.get(chat_id, []))
        add_to_history(chat_id, "user", f"[Immagine] {didascalia}")
        add_to_history(chat_id, "assistant", risposta)
        bot.sendMessage(chat_id, risposta)
        os.remove(temp_img)        

    # Gestione Vocali o File Audio
    elif content_type in ['voice', 'audio']:
        bot.sendMessage(chat_id, "Elaboro audio...")
        
        # Recupera il file_id corretto
        file_id = msg[content_type]['file_id']
        file_info = bot.getFile(file_id)
        
        # Definisci un percorso temporaneo per il file
        temp_filename = f"temp_{chat_id}_{int(time.time())}.wav"
        
        try:
            # Scarica il file da Telegram
            bot.download_file(file_id, temp_filename)
            
            # Trascrivi
            testo_trascritto = trascrivi_audio(temp_filename)
            print(testo_trascritto)
            # Rispondi all'utente
            if testo_trascritto:
                bot.sendMessage(chat_id, "Pensando a una risposta...")
                risposta = llm.ask(testo_trascritto)
                bot.sendMessage(chat_id, risposta)
            else:
                bot.sendMessage(chat_id, "Non sono riuscito a estrarre testo da questo audio.")
                
        except Exception as e:
            bot.sendMessage(chat_id, f"Si è verificato un errore: {e}")
        
        finally:
            # Pulizia: rimuove il file temporaneo
            if os.path.exists(temp_filename):
                os.remove(temp_filename)



print("Bot started...")
MessageLoop(bot, handle).run_as_thread()

if __name__ == "__main__":
    while True:
        time.sleep(1000)