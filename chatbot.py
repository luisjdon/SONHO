import random

class ConversationalAI:
    def __init__(self):
        # Banco de respostas pré-definidas
        self.responses = {
            "saudacao": ["Olá! Como posso te ajudar hoje?", "Oi! Tudo bem com você?", "Boa noite! Em que posso ajudar?"],
            "nome": ["Meu nome é Sonho, prazer!", "Eu sou Sonho, sua assistente virtual.", "Sonho, à sua disposição!"],
            "como_esta": ["Estou bem, obrigada por perguntar! E você?", "Estou ótima, pronta para te ajudar!", "Muito bem, e você?"],
            "desconhecido": ["Desculpe, não entendi. Pode reformular?", "Ainda estou aprendendo. Pode explicar melhor?", "Não sei muito sobre isso ainda, mas você pode me ensinar!"]
        }

    def identify_intent(self, user_input):
        """Identifica a intenção do usuário com base na entrada."""
        user_input = user_input.lower()
        
        # Intenções simples baseadas em palavras-chave
        if any(word in user_input for word in ["oi", "olá", "boa noite", "bom dia", "boa tarde"]):
            return "saudacao"
        elif any(word in user_input for word in ["qual o seu nome", "quem é você", "seu nome"]):
            return "nome"
        elif any(word in user_input for word in ["como você está", "tudo bem", "como vai"]):
            return "como_esta"
        else:
            return "desconhecido"

    def get_response(self, intent):
        """Seleciona uma resposta com base na intenção."""
        return random.choice(self.responses[intent])

    def chat(self):
        """Inicia o loop de conversa."""
        print("Sonho: Olá! Eu sou Sonho, sua assistente virtual. Como posso te ajudar hoje?")
        while True:
            user_input = input("Você: ")
            
            # Comando para sair
            if user_input.lower() in ["sair", "tchau", "até logo"]:
                print("Sonho: Até logo! Foi um prazer conversar com você.")
                break
            
            # Identificar intenção e gerar resposta
            intent = self.identify_intent(user_input)
            response = self.get_response(intent)
            print(f"Sonho: {response}")

# Executar a IA de conversação
if __name__ == "__main__":
    bot = ConversationalAI()
    bot.chat()