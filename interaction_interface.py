from chatbot_gpt import GPTChatbot

class InteractionInterface:
    def __init__(self):
        self.chatbot = GPTChatbot()  # Instanciando o chatbot GPT

    def start_interaction_loop(self):
        """Inicia o loop de interação com o chatbot GPT."""
        print("Bem-vindo ao Chatbot Sonho!")
        self.chatbot.chat()  # Usando o loop de conversa do GPTChatbot