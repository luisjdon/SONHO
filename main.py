import os
from knowledge_base import KnowledgeBase
from learning_engine import LearningEngine
from interaction_interface import InteractionInterface
from calculation_module import CalculationModule

def main():
    # Garantir que o diretório de dados exista
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Inicializar componentes
    knowledge_base = KnowledgeBase('data/knowledge.json')
    calculation_module = CalculationModule()
    learning_engine = LearningEngine(knowledge_base, calculation_module)
    interface = InteractionInterface(learning_engine)
    
    print("=== IA Aprendente ===")
    print("Digite 'sair' para encerrar a aplicação.")
    print("Digite 'aprender: [tópico] = [informação]' para ensinar algo à IA.")
    print("Digite 'calcular: [expressão]' para realizar cálculos.")
    print("Digite qualquer outra coisa para conversar com a IA.")
    print("============================")
    
    # Iniciar loop de interação
    interface.start_interaction_loop()

if __name__ == "__main__":
    main()