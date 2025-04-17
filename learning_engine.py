class LearningEngine:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def process_input(self, user_input):
        # Aqui você poderia integrar o GPT para processar as entradas
        # Exemplo: Atualizar o conhecimento com base em novas informações
        if "ensinar" in user_input.lower():
            # Lógica para ensinar algo novo
            fact = user_input.split("ensinar:")[-1].strip()
            self.knowledge_base.add_fact("informacoes", fact)
            return "Obrigado por me ensinar algo novo!"
        else:
            return "Estou aprendendo com você!"