import json
import os
from datetime import datetime, timedelta
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

class KnowledgeBase:
    def __init__(self, file_path):
        self.file_path = file_path
        
        # Ensure NLTK resources are downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.lemmatizer = WordNetLemmatizer()
        self.knowledge = self._load_knowledge()
        
        # Initialize vocabulary correctly
        if "vocabulary" not in self.knowledge or not isinstance(self.knowledge["vocabulary"], dict):
            self.knowledge["vocabulary"] = {}
            
        # Initialize personality if not exists
        if "personality" not in self.knowledge:
            self.knowledge["personality"] = {
                "name": "Sonho",
                "traits": ["curiosa", "amigável", "prestativa"],
                "interests": ["aprendizado", "conversas", "conhecimento"],
                "greeting": "Olá! Sou Sonho, sua IA aprendente. Como posso ajudar hoje?"
            }

    def _load_knowledge(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as file:
                    knowledge = json.load(file)
                    return knowledge
            except json.JSONDecodeError:
                print("Arquivo de conhecimento corrompido. Criando novo arquivo.")
        
        # Return default if file doesn't exist or is corrupted
        return {
            "facts": {},
            "conversations": [],
            "last_updated": str(datetime.now()),
            "relationships": {},
            "vocabulary": {},
            "code_knowledge": {},
            "conversation_memory": [],
            "personality": {}
        }

    def save_knowledge(self):
        self.knowledge["last_updated"] = str(datetime.now())
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, 'w', encoding='utf-8') as file:
            json.dump(self.knowledge, file, ensure_ascii=False, indent=4)

    def add_fact(self, topic, information):
        """Add a new fact to the knowledge base"""
        topic = topic.lower().strip()
        
        if topic not in self.knowledge["facts"]:
            self.knowledge["facts"][topic] = []

        # Check if information already exists
        for existing_fact in self.knowledge["facts"][topic]:
            if isinstance(existing_fact, dict):
                if existing_fact.get("text", "") == information:
                    return False
            elif existing_fact == information:
                return False

        # Add new fact with timestamp
        fact_entry = {
            "text": information,
            "timestamp": str(datetime.now()),
            "confidence": 1.0
        }
        
        self.knowledge["facts"][topic].append(fact_entry)
        self._extract_keywords(topic, information)
        self.save_knowledge()
        return True

    def _extract_keywords(self, topic, information):
        """Extract keywords from information and update vocabulary"""
        # Tokenize the text
        words = word_tokenize(information.lower())
        
        # Get stopwords for filtering
        stop_words = set(stopwords.words('portuguese') + stopwords.words('english'))
        
        # Filter out stopwords and punctuation
        words = [
            self.lemmatizer.lemmatize(word) for word in words
            if word not in stop_words and word not in string.punctuation
        ]

        # Update vocabulary
        for word in words:
            if word not in self.knowledge["vocabulary"]:
                self.knowledge["vocabulary"][word] = {"topics": [], "count": 0}

            if topic not in self.knowledge["vocabulary"][word]["topics"]:
                self.knowledge["vocabulary"][word]["topics"].append(topic)

            self.knowledge["vocabulary"][word]["count"] += 1

    def search_knowledge(self, query):
        """Search knowledge base for relevant information"""
        query = query.lower().strip()
        results = []
        
        # Tokenize the query
        query_words = word_tokenize(query)
        
        # Filter out stopwords and punctuation
        query_words = [
            self.lemmatizer.lemmatize(word) for word in query_words
            if word not in stopwords.words('portuguese') and
               word not in stopwords.words('english') and
               word not in string.punctuation
        ]

        # Calculate relevance scores for topics
        topic_scores = {}
        for word in query_words:
            if word in self.knowledge["vocabulary"]:
                for topic in self.knowledge["vocabulary"][word]["topics"]:
                    topic_scores[topic] = topic_scores.get(topic, 0) + self.knowledge["vocabulary"][word]["count"]

        # Sort topics by relevance
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)

        # Get facts from relevant topics
        for topic, score in sorted_topics[:5]:
            if topic in self.knowledge["facts"]:
                for fact in self.knowledge["facts"][topic]:
                    if isinstance(fact, dict):
                        fact_text = fact.get("text", "")
                        results.append({"topic": topic, "fact": fact_text, "score": score})
                    else:
                        results.append({"topic": topic, "fact": fact, "score": score})

        # Sort results by relevance score
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:5]

    def add_conversation(self, user_input, ai_response):
        """Add a conversation exchange to history"""
        # Add to regular conversations list
        if len(self.knowledge["conversations"]) >= 100:
            self.knowledge["conversations"].pop(0)

        self.knowledge["conversations"].append({
            "user": user_input,
            "ai": ai_response,
            "timestamp": str(datetime.now())
        })

        # Add to conversation memory (for context awareness)
        if len(self.knowledge["conversation_memory"]) >= 10:
            self.knowledge["conversation_memory"].pop(0)
            
        self.knowledge["conversation_memory"].append({
            "user": user_input,
            "ai": ai_response,
            "timestamp": str(datetime.now())
        })

        self.save_knowledge()

    def get_conversation_memory(self):
        """Get recent conversations for context"""
        return self.knowledge["conversation_memory"]

    def add_code_knowledge(self, language, concept, explanation):
        """Add programming knowledge to the knowledge base"""
        if "code_knowledge" not in self.knowledge:
            self.knowledge["code_knowledge"] = {}
            
        if language not in self.knowledge["code_knowledge"]:
            self.knowledge["code_knowledge"][language] = {}
            
        self.knowledge["code_knowledge"][language][concept] = {
            "explanation": explanation,
            "timestamp": str(datetime.now())
        }
        
        self.save_knowledge()
        return True
        
    def get_code_knowledge(self, language, concept=None):
        """Get programming knowledge from the knowledge base"""
        if "code_knowledge" not in self.knowledge:
            return None
            
        if language not in self.knowledge["code_knowledge"]:
            return None
            
        if concept:
            return self.knowledge["code_knowledge"][language].get(concept)
        else:
            return self.knowledge["code_knowledge"][language]
            
    def analyze_code(self, language, code):
        """Basic code analysis using known concepts"""
        if "code_knowledge" not in self.knowledge or language not in self.knowledge["code_knowledge"]:
            return "Ainda não aprendi sobre esta linguagem de programação."
            
        known_concepts = self.knowledge["code_knowledge"][language].keys()
        analysis = f"Analisando código {language}:\n"
        
        for concept in known_concepts:
            if concept.lower() in code.lower():
                analysis += f"- Identificado: {concept}\n"
                explanation = self.knowledge["code_knowledge"][language][concept]["explanation"]
                analysis += f"  {explanation}\n"
                
        if analysis == f"Analisando código {language}:\n":
            analysis += "Não identifiquei conceitos que conheço neste código."
            
        return analysis
        
    def update_personality(self, trait, value):
        """Update personality traits"""
        if "personality" not in self.knowledge:
            self.knowledge["personality"] = {}
        
        self.knowledge["personality"][trait] = value
        self.save_knowledge()
        
    def get_personality(self):
        """Get personality information"""
        if "personality" not in self.knowledge:
            self.knowledge["personality"] = {
                "name": "Sonho",
                "traits": ["curiosa", "amigável", "prestativa"],
                "interests": ["aprendizado", "conversas", "conhecimento"],
                "greeting": "Olá! Sou Sonho, sua IA aprendente. Como posso ajudar hoje?"
            }
        
        return self.knowledge["personality"]

    def get_all_topics(self):
        """Get all known topics"""
        return list(self.knowledge["facts"].keys())

    def get_facts_for_topic(self, topic):
        """Get all facts for a specific topic"""
        topic = topic.lower().strip()
        if topic in self.knowledge["facts"]:
            facts = []
            for fact in self.knowledge["facts"][topic]:
                if isinstance(fact, dict):
                    facts.append(fact["text"])
                else:
                    facts.append(fact)
            return facts
        return []