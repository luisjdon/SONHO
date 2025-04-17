import json
from datetime import datetime

class KnowledgeBase:
    def __init__(self, file_path="knowledge_base.json"):
        self.file_path = file_path
        self.knowledge = {"facts": {}, "conversations": []}
        self.load_knowledge()

    def load_knowledge(self):
        try:
            with open(self.file_path, "r") as file:
                self.knowledge = json.load(file)
        except FileNotFoundError:
            self.save_knowledge()

    def save_knowledge(self):
        with open(self.file_path, "w") as file:
            json.dump(self.knowledge, file, indent=4)

    def add_fact(self, category, fact):
        if category not in self.knowledge["facts"]:
            self.knowledge["facts"][category] = []
        self.knowledge["facts"][category].append(fact)
        self.save_knowledge()

    def store_conversation(self, user_input, ai_response):
        self.knowledge["conversations"].append({
            "user": user_input,
            "ai": ai_response,
            "timestamp": str(datetime.now())
        })
        self.save_knowledge()

    def recall_information(self, query):
        # Buscar informações relacionadas ao que foi perguntado
        results = [fact for fact in self.knowledge["facts"].get("informacoes", []) if query in fact]
        return results