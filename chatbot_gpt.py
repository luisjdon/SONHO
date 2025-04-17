from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class GPTChatbot:
    def __init__(self, model_name="EleutherAI/gpt-neo-1.3B"):
        print("Carregando modelo... Isso pode levar alguns minutos.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history = ""  # Histórico da conversa para contexto

    def generate_response(self, user_input):
        # Adicionar entrada do usuário ao histórico
        self.chat_history += f"Usuário: {user_input}\nSonho: "

        # Configurar o token de preenchimento
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenizar o histórico da conversa com attention_mask
        inputs = self.tokenizer(
            self.chat_history, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024, 
            padding=True
        )

        # Gerar resposta
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,  # Passar o attention_mask explicitamente
            max_length=1024,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decodificar a resposta gerada
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extrair apenas a resposta mais recente
        response = response.split("Sonho:")[-1].strip()

        # Adicionar a resposta ao histórico
        self.chat_history += response + "\n"

        return response

    def chat(self):
        print("Sonho: Olá! Eu sou Sonho, sua assistente virtual. Como posso te ajudar hoje?")
        while True:
            user_input = input("Você: ")
            if user_input.lower() in ["sair", "tchau", "adeus"]:
                print("Sonho: Até logo! Foi um prazer conversar com você.")
                break

            # Gerar resposta para a entrada do usuário
            try:
                response = self.generate_response(user_input)
                print(f"Sonho: {response}")
            except Exception as e:
                print(f"Sonho: Ocorreu um erro ao gerar a resposta. Detalhes: {e}")