class InteractionInterface:
    def __init__(self, learning_engine):
        self.learning_engine = learning_engine
    
    def start_interaction_loop(self):
        """Inicia o loop de interação com o usuário."""
        while True:
            try:
                user_input = input("\nVocê: ")
                
                # Verificar comando de saída
                if user_input.lower() == 'sair':
                    print("Até logo! Obrigado por conversar comigo.")
                    break
                
                # Processar a entrada e obter resposta
                ai_response = self.learning_engine.process_input(user_input)
                
                # Exibir resposta
                print(f"IA: {ai_response}")
                
                # Registrar a conversa
                self.learning_engine.knowledge_base.add_conversation(user_input, ai_response)
                
            except KeyboardInterrupt:
                print("\nEncerrando o programa...")
                break
            except Exception as e:
                print(f"Ocorreu um erro: {str(e)}")
    
    def get_ai_response(self, user_input):
        """Obtém uma resposta da IA para uma entrada específica."""
        return self.learning_engine.process_input(user_input)