# main.py - Arquivo principal

import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import PyPDF2
import io
from werkzeug.utils import secure_filename

# Configurar NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Classe de Base de Conhecimento
class KnowledgeBase:
    def __init__(self, file_path='knowledge.json'):
        self.file_path = file_path
        self.knowledge = self._load_knowledge()

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

        # Garante a estrutura correta
        if "vocabulary" not in knowledge:
            knowledge["vocabulary"] = {}
            
        return knowledge

    def _create_empty_knowledge(self):
        return {
            "facts": {},
            "conversations": [],
            "documents": {},
            "last_updated": str(datetime.now()),
            "vocabulary": {}
        }

    def save_knowledge(self):
        with open(self.file_path, 'w', encoding='utf-8') as file:
            json.dump(self.knowledge, file, ensure_ascii=False, indent=4)

    def add_fact(self, topic, information):
        topic = topic.lower().strip()
        if topic not in self.knowledge["facts"]:
            self.knowledge["facts"][topic] = []

        # Verifica se a informação já existe
        if information not in self.knowledge["facts"][topic]:
            self.knowledge["facts"][topic].append(information)
            self._extract_keywords(topic, information)
            self.save_knowledge()
            return True
        return False

    def _extract_keywords(self, topic, text):
        # Tokeniza e processa o texto para extrair palavras-chave
        words = nltk.word_tokenize(text.lower())
        stop_words = set(nltk.corpus.stopwords.words('portuguese') + 
                         nltk.corpus.stopwords.words('english'))
        
        # Filtra palavras significativas
        meaningful_words = [
            word for word in words
            if word not in stop_words and len(word) > 2
        ]
        
        # Adiciona ao vocabulário
        for word in meaningful_words:
            if word not in self.knowledge["vocabulary"]:
                self.knowledge["vocabulary"][word] = {"topics": [], "count": 0}
            
            if topic not in self.knowledge["vocabulary"][word]["topics"]:
                self.knowledge["vocabulary"][word]["topics"].append(topic)
            
            self.knowledge["vocabulary"][word]["count"] += 1

    def search_knowledge(self, query):
        query = query.lower().strip()
        results = []
        
        # Busca direta por tópico
        for topic, facts in self.knowledge["facts"].items():
            if query in topic:
                for fact in facts:
                    results.append({"topic": topic, "fact": fact, "score": 10})
                    
        # Busca por palavras-chave
        query_words = nltk.word_tokenize(query)
        for word in query_words:
            if word in self.knowledge["vocabulary"]:
                for topic in self.knowledge["vocabulary"][word]["topics"]:
                    score = self.knowledge["vocabulary"][word]["count"]
                    for fact in self.knowledge["facts"].get(topic, []):
                        results.append({"topic": topic, "fact": fact, "score": score})
        
        # Elimina duplicatas e ordena por relevância
        unique_results = {}
        for item in results:
            key = f"{item['topic']}:{item['fact']}"
            if key not in unique_results or item['score'] > unique_results[key]['score']:
                unique_results[key] = item
                
        sorted_results = sorted(unique_results.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:5]  # Retorna os 5 mais relevantes

    def add_conversation(self, user_input, ai_response):
        # Limita o histórico a 100 conversas
        if len(self.knowledge["conversations"]) >= 100:
            self.knowledge["conversations"].pop(0)
            
        self.knowledge["conversations"].append({
            "user": user_input,
            "ai": ai_response,
            "timestamp": str(datetime.now())
        })
        
        self.save_knowledge()
        
    def add_document(self, doc_name, content):
        """Adiciona um documento à base de conhecimento"""
        if "documents" not in self.knowledge:
            self.knowledge["documents"] = {}
            
        self.knowledge["documents"][doc_name] = content
        
        # Extrai conhecimento do documento
        topics = self._extract_topics_from_document(content)
        for topic, facts in topics.items():
            for fact in facts:
                self.add_fact(topic, fact)
                
        self.save_knowledge()
        return True
        
    def _extract_topics_from_document(self, content):
        """Extrai tópicos e fatos de um documento"""
        topics = {}
        
        # Divide o conteúdo em parágrafos
        paragraphs = content.split('\n\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            # Tenta identificar um tópico para o parágrafo
            sentences = nltk.sent_tokenize(paragraph)
            if not sentences:
                continue
                
            # Usa a primeira sentença como possível indicador de tópico
            first_sent = sentences[0].lower()
            words = nltk.word_tokenize(first_sent)
            
            # Remove stopwords para identificar possíveis tópicos
            stop_words = set(nltk.corpus.stopwords.words('portuguese') + 
                            nltk.corpus.stopwords.words('english'))
            keywords = [w for w in words if w not in stop_words and len(w) > 3]
            
            # Se encontrou palavras-chave, usa a primeira como tópico
            topic = "geral"
            if keywords:
                topic = keywords[0]
                
            # Adiciona o parágrafo como um fato sob este tópico
            if topic not in topics:
                topics[topic] = []
            topics[topic].append(paragraph)
            
        return topics

# Classe do Chatbot GPT
class SonhoChatbot:
    def __init__(self, knowledge_base, model_name="EleutherAI/gpt-neo-1.3B"):
        self.knowledge_base = knowledge_base
        print("Carregando modelo... Isso pode levar alguns minutos.")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.chat_history = ""
            print("Modelo carregado com sucesso!")
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            raise
            
    def generate_response(self, user_input):
        # Pesquisa na base de conhecimento
        knowledge_results = self.knowledge_base.search_knowledge(user_input)
        
        # Prepara o contexto com o conhecimento relevante
        context = ""
        if knowledge_results:
            context = "Conhecimento relevante:\n"
            for item in knowledge_results:
                context += f"- {item['topic']}: {item['fact']}\n"
            context += "\n"
        
        # Formata a entrada para o modelo
        prompt = f"{context}Histórico recente:\n{self.chat_history[-1000:] if len(self.chat_history) > 1000 else self.chat_history}\nUsuário: {user_input}\nSonho: "
        
        # Configurar tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Tokeniza a entrada
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024, 
            padding=True
        )
        
        # Gera a resposta
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=1024,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decodifica a resposta
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extrai apenas a parte da resposta do assistente
        response = full_response.split("Sonho:")[-1].strip()
        
        # Adiciona ao histórico
        self.chat_history += f"Usuário: {user_input}\nSonho: {response}\n\n"
        
        # Salva a conversa na base de conhecimento
        self.knowledge_base.add_conversation(user_input, response)
        
        return response
    
    def process_command(self, user_input):
        """Processa comandos especiais"""
        # Comando para ensinar algo
        if user_input.lower().startswith("aprender:"):
            try:
                # Formato esperado: "aprender: tópico = informação"
                parts = user_input[9:].split("=", 1)
                if len(parts) == 2:
                    topic = parts[0].strip()
                    info = parts[1].strip()
                    self.knowledge_base.add_fact(topic, info)
                    return f"Obrigado! Aprendi sobre '{topic}': {info}"
                else:
                    return "Formato inválido. Use: aprender: tópico = informação"
            except Exception as e:
                return f"Erro ao processar comando de aprendizado: {str(e)}"
                
        # Resposta normal
        return self.generate_response(user_input)
        
    def process_pdf(self, file_stream, filename):
        """Processa um arquivo PDF e extrai conhecimento"""
        try:
            # Extrai texto do PDF
            pdf_reader = PyPDF2.PdfReader(file_stream)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
                
            # Adiciona à base de conhecimento
            self.knowledge_base.add_document(filename, text)
            
            return f"Arquivo '{filename}' processado com sucesso! Aprendi {len(text.split())} palavras deste documento."
        except Exception as e:
            return f"Erro ao processar o PDF: {str(e)}"

# Criando a aplicação Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Inicializa a base de conhecimento e o chatbot
knowledge_base = KnowledgeBase('knowledge.json')
chatbot = SonhoChatbot(knowledge_base)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({"response": "Por favor, envie uma mensagem."})
    
    # Verifica se é um comando especial
    if message.lower().startswith(("aprender:", "ensinar:")):
        response = chatbot.process_command(message)
    else:
        response = chatbot.generate_response(message)
        
    return jsonify({"response": response})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"response": "Nenhum arquivo enviado"})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"response": "Nenhum arquivo selecionado"})
        
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Processa o arquivo conforme o tipo
        if filename.lower().endswith('.pdf'):
            with open(file_path, 'rb') as f:
                response = chatbot.process_pdf(f, filename)
            return jsonify({"response": response})
        else:
            return jsonify({"response": f"Arquivo '{filename}' recebido, mas o formato não é suportado. Por favor, envie arquivos PDF."})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)