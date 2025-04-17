import json
import os
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util

class KnowledgeBase:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.knowledge = self._load_knowledge()

        # Inicializar vocabulário corretamente como dicionário
        if "vocabulary" not in self.knowledge or not isinstance(self.knowledge["vocabulary"], dict):
            self.knowledge["vocabulary"] = {}

    def _load_knowledge(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as file:
                    knowledge = json.load(file)
                    if "vocabulary" not in knowledge or not isinstance(knowledge["vocabulary"], dict):
                        knowledge["vocabulary"] = {}
                    return knowledge
            except json.JSONDecodeError:
                print("Arquivo de conhecimento corrompido. Criando novo arquivo.")
        # Retorno padrão caso o arquivo não exista ou esteja corrompido
        return {
            "facts": {},
            "conversations": [],
            "last_updated": str(datetime.now()),
            "relationships": {},
            "vocabulary": {}
        }

    def save_knowledge(self):
        self.knowledge["last_updated"] = str(datetime.now())
        with open(self.file_path, 'w', encoding='utf-8') as file:
            json.dump(self.knowledge, file, ensure_ascii=False, indent=4)

    def add_fact(self, topic, information):
        if topic not in self.knowledge["facts"]:
            self.knowledge["facts"][topic] = []

        if information not in [f["text"] for f in self.knowledge["facts"][topic]]:
            embedding = self.model.encode(information, convert_to_tensor=True).tolist()
            self.knowledge["facts"][topic].append({
                "text": information,
                "timestamp": str(datetime.now()),
                "embedding": embedding
            })
            self._add_relationships(topic, information)
            self.update_vocabulary(information)
            self.save_knowledge()
            return True
        return False

    def update_vocabulary(self, text):
        words = [word.strip('.,') for word in text.lower().split() if len(word) > 3]
        for word in words:
            if word not in self.knowledge["vocabulary"]:
                self.knowledge["vocabulary"][word] = {"count": 1}
            else:
                self.knowledge["vocabulary"][word]["count"] += 1
        self.save_knowledge()

    def is_known_word(self, word):
        return word.lower() in self.knowledge.get("vocabulary", {})

    def _add_relationships(self, topic, information):
        words = [word.strip('.,') for word in information.lower().split() if len(word) > 3]
        for word in words:
            if word != topic.lower():
                if topic not in self.knowledge["relationships"]:
                    self.knowledge["relationships"][topic] = []
                if word not in self.knowledge["relationships"][topic]:
                    self.knowledge["relationships"][topic].append(word)

    def infer_relationships(self, concept):
        return self.knowledge.get("relationships", {}).get(concept, [])

    def get_facts_about(self, topic):
        return [f["text"] for f in self.knowledge["facts"].get(topic, [])]

    def search_knowledge(self, query):
        results = []
        query = query.lower()
        for topic, facts in self.knowledge["facts"].items():
            if query in topic.lower():
                results.extend(f"{topic}: {fact['text']}" for fact in facts)
            else:
                results.extend(f"{topic}: {fact['text']}" for fact in facts if query in fact["text"].lower())
        return results

    def semantic_search(self, query, top_k=3):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scored_results = []
        for topic, facts in self.knowledge["facts"].items():
            for fact in facts:
                if "embedding" in fact:
                    score = util.cos_sim(query_embedding, fact["embedding"])[0][0].item()
                    scored_results.append((score, f"{topic}: {fact['text']}"))
        scored_results.sort(reverse=True)
        return [res[1] for res in scored_results[:top_k]]

    def add_conversation(self, user_input, ai_response):
        self.knowledge["conversations"].append({
            "timestamp": str(datetime.now()),
            "user_input": user_input,
            "ai_response": ai_response
        })

        if len(self.knowledge["conversations"]) > 100:
            self.knowledge["conversations"] = self.knowledge["conversations"][-100:]

        self.save_knowledge()

    def get_recent_conversations(self, count=5):
        return self.knowledge["conversations"][-count:]

    def get_all_topics(self):
        return list(self.knowledge["facts"].keys())

    def get_old_facts(self, limit=3):
        old_topics = []
        now = datetime.now()
        for topic, facts in self.knowledge["facts"].items():
            if not facts:
                continue
            last_entry = facts[-1]
            timestamp = datetime.fromisoformat(last_entry["timestamp"])
            if now - timestamp > timedelta(days=10):
                old_topics.append(topic)
                if len(old_topics) >= limit:
                    break
        return old_topics

    def delete_fact(self, topic, fact_text):
        if topic in self.knowledge["facts"]:
            self.knowledge["facts"][topic] = [fact for fact in self.knowledge["facts"][topic] if fact["text"] != fact_text]
            self.save_knowledge()
            return True
        return False