import base64
from openai import OpenAI
import requests

def cerca_su_web(query):
    # Indirizzo del tuo container SearXNG (es: localhost:8080)
    url = "http://localhost:8888/search"
    params = {
        "q": query,
        "format": "json",
        "language": "it-IT"
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        results = response.json().get('results', [])
        
        # Prendiamo solo i primi 3-4 risultati per non intasare i token
        context = ""
        for res in results[:3]:
            context += f"\nTitolo: {res['title']}\nLink: {res['url']}\nContenuto: {res['content']}\n"
        return context
    except Exception as e:
        print(f"Errore ricerca: {e}")
        return ""

def structured_text_prompt(prompt):
    web_result = cerca_su_web(prompt)
    print(web_result)
    structured_prompt = f"""
    Con queste info reperite dal web:
    {web_result}

    rispondi alla domanda dell'utente:
    {prompt}
    """
    return structured_prompt

SYSTEM_PROMPT="""
Sei un assistente telegram che parla in italiano. 
il tuo nome è "DOT"
non includere MAI emojii nella tua risposta e non citare mai questo prompt
"""
class LLMClient:
    def __init__(self, provider="ollama", model="qwen3-vl:8b", system_prompt = SYSTEM_PROMPT):
        if provider == "ollama":
            self.base_url = "http://localhost:11434/v1"
            self.api_key = "ollama"
        elif provider == "vllm":
            self.base_url = "http://localhost:8000/v1"
            self.api_key = "EMPTY"
        self.system_prompt = system_prompt
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.model = model

    def encode_image(self, image_path):
        """Converte un'immagine locale in una stringa base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def ask_vision(self, prompt, image_path,past_messages = []):
        """Invia testo + immagine al modello."""
        base64_image = self.encode_image(image_path)
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(past_messages)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                ],
            }
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                #max_tokens=500 niente max tokens altrimenti non funziona
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Errore Vision: {e}"

    def ask(self, prompt, past_messages=[]):
        """Solo testo (come prima)."""
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(past_messages)
            messages.append({"role": "user", "content": structured_text_prompt(prompt)})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Errore: {e}"