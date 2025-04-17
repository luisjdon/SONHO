from flask import Flask, render_template, request, jsonify
import os
import json
import re
import random
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from datetime import datetime

app = Flask(__name__)

# Configuração inicial do NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Classe de Base de Conhecimento
class KnowledgeBase:
    def __init__(self, file_path):
        self.file_path = file_path
        self.knowledge = self._load_knowledge()
        self.lemmatizer = WordNetLemmatizer()

    def _load_knowledge(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as file:
                    knowledge = json.load(file)
            except json.JSONDecodeError:
                print("Arquivo corrompido. Criando novo.")
                knowledge = self._create_empty_knowledge()
        else:
            knowledge = self._create_empty_knowledge()

        # Conversão de set para dict se necessário
        if isinstance(knowledge.get("vocabulary"), set):
            knowledge["vocabulary"] = {}

        # Se o vocabulário não existir, cria como dicionário
        if "vocabulary" not in knowledge:
            knowledge["vocabulary"] = {}

        return knowledge

    def _create_empty_knowledge(self):
        return {
            "facts": {},
            "conversations": [],
            "last_updated": str(datetime.now()),
            "relationships": {},
            "vocabulary": {}
        }

    def save_knowledge(self):
        with open(self.file_path, 'w', encoding='utf-8') as file:
            json.dump(self.knowledge, file, ensure_ascii=False, indent=4)

    def add_fact(self, topic, information):
        topic = topic.lower().strip()
        if topic not in self.knowledge["facts"]:
            self.knowledge["facts"][topic] = []

        if information not in self.knowledge["facts"][topic]:
            self.knowledge["facts"][topic].append(information)
            self._extract_keywords(topic, information)
            self.save_knowledge()
            return True
        return False

    def _extract_keywords(self, topic, information):
        words = word_tokenize(information.lower())
        stop_words = set(stopwords.words('portuguese') + stopwords.words('english'))
        words = [
            self.lemmatizer.lemmatize(word) for word in words
            if word not in stop_words and word not in string.punctuation
        ]

        for word in words:
            if word not in self.knowledge["vocabulary"]:
                self.knowledge["vocabulary"][word] = {"topics": [], "count": 0}

            if topic not in self.knowledge["vocabulary"][word]["topics"]:
                self.knowledge["vocabulary"][word]["topics"].append(topic)

            self.knowledge["vocabulary"][word]["count"] += 1

    def search_knowledge(self, query):
        query = query.lower().strip()
        results = []
        query_words = word_tokenize(query)
        query_words = [
            self.lemmatizer.lemmatize(word) for word in query_words
            if word not in stopwords.words('portuguese') and
               word not in stopwords.words('english') and
               word not in string.punctuation
        ]

        topic_scores = {}
        for word in query_words:
            if word in self.knowledge["vocabulary"]:
                for topic in self.knowledge["vocabulary"][word]["topics"]:
                    topic_scores[topic] = topic_scores.get(topic, 0) + self.knowledge["vocabulary"][word]["count"]

        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)

        for topic, score in sorted_topics[:5]:
            for fact in self.knowledge["facts"].get(topic, []):
                results.append({"topic": topic, "fact": fact, "score": score})

        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:5]

    def add_conversation(self, user_input, ai_response):
        if len(self.knowledge["conversations"]) >= 100:
            self.knowledge["conversations"].pop(0)

        self.knowledge["conversations"].append({
            "user": user_input,
            "ai": ai_response
        })

        self.save_knowledge()

# Inicializa a base
knowledge_base = KnowledgeBase('knowledge.json')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message')

    response_data = knowledge_base.search_knowledge(message)

    if response_data:
        response = f"Eu encontrei algo sobre isso: {response_data[0]['fact']}"
    else:
        response = "Não sei sobre isso. Você gostaria de me ensinar? Se sim, por favor, diga o tópico e a informação."

        if ':' in message:
            topic, fact = message.split(":", 1)
            topic = topic.strip()
            fact = fact.strip()
            knowledge_base.add_fact(topic, fact)
            response = f"Obrigado por me ensinar sobre '{topic}'. Agora sei sobre isso!"

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
