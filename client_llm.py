import base64
import requests
from openai import OpenAI

# --- CONFIGURAZIONE E UTILITY ---

def cerca_su_web(query):
    """Interroga SearXNG e restituisce un contesto testuale."""
    if not query or query.lower() == "nessuna":
        return "Nessuna ricerca necessaria."
        
    url = "http://localhost:8888/search"
    params = {
        "q": query,
        "format": "json",
        "language": "it-IT"
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        results = response.json().get('results', [])
        
        context = ""
        for res in results[:3]:
            context += f"\nFonte: {res['title']}\nContenuto: {res['content']}\n"
        return context if context else "Nessun risultato trovato sul web."
    except Exception as e:
        print(f"Errore ricerca: {e}")
        return "Errore durante la connessione al web."

SYSTEM_PROMPT = """
Sei un assistente telegram che parla in italiano. 
Il tuo nome è "DOT".
NON includere MAI emoji nella tua risposta.
Usa i dati forniti dal web per essere preciso.
Se non trovi informazioni, ammettilo senza inventare.
"""

# --- CLASSE CLIENT AGGIORNATA ---
#qwen3-vl:8b
class LLMClient:
    def __init__(self, provider="ollama", model="gemma4:e2b"):
        if provider == "ollama":
            self.base_url = "http://192.168.1.54:11434/v1"
            self.api_key = "ollama"
        elif provider == "vllm":
            self.base_url = "http://localhost:8000/v1"
            self.api_key = "EMPTY"
            
        self.system_prompt = SYSTEM_PROMPT
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.model = model

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def ask_vision(self, prompt, image_path, past_messages=[]):
        """Doppio passaggio: Identificazione -> Ricerca -> Risposta finale."""
        base64_image = self.encode_image(image_path)
        
        # --- PASSAGGIO 1: Generazione Query di Ricerca ---
        # Chiediamo al modello di identificare cosa cercare basandosi sull'immagine
        search_query_prompt = f"Analizza l'immagine e scrivi solo 3-4 parole chiave per una ricerca su Google che aiuti a rispondere a: {prompt}. Se non serve cercare, scrivi 'Nessuna'."
        
        try:
            temp_messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": search_query_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
            query_res = self.client.chat.completions.create(
                model=self.model,
                messages=temp_messages,
            )
            query = query_res.choices[0].message.content.strip().replace('"', '')
            print(f"Query generata (Vision): {query}")
            
            # --- WEB SEARCH ---
            web_context = cerca_su_web(query)
            
            # --- PASSAGGIO 2: Risposta Finale ---
            final_prompt = f"""
            CONTESTO WEB:
            {web_context}

            DOMANDA UTENTE:
            {prompt}
            
            Rispondi usando i dettagli dell'immagine e le info del web.
            """
            
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(past_messages)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": final_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            })
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"Errore Vision doppio passaggio: {e}"

    def ask(self, prompt, past_messages=[]):
        """Doppio passaggio testuale: Refinement Query -> Ricerca -> Risposta."""
        try:
            # --- PASSAGGIO 1: Ottimizzazione Query ---
            """refine_prompt = f"Trasforma questa richiesta in una query di ricerca efficace per SearXNG. Scrivi solo la query: {prompt}"
            query_res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": refine_prompt}],
            )
            query = query_res.choices[0].message.content.strip()
            print(f"Query generata (Text): {query}")"""

            # --- WEB SEARCH ---
            web_context = cerca_su_web(prompt)

            # --- PASSAGGIO 2: Risposta Finale ---
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(past_messages)
            
            final_content = f"Dati Web:\n{web_context}\n\nDomanda: {prompt}"
            messages.append({"role": "user", "content": final_content})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Errore Testo doppio passaggio: {e}"