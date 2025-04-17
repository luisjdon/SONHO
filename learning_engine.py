import re
import random
from datetime import datetime
import spacy
import nltk

# Garantir que temos os recursos necessários
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class LearningEngine:
    def __init__(self, knowledge_base, calculation_module):
        self.knowledge_base = knowledge_base
        self.calculation_module = calculation_module
        
        # Carregar modelo spaCy para análise linguística
        try:
            self.nlp_spacy = spacy.load("pt_core_news_md")
        except:
            try:
                import os
                os.system("python -m spacy download pt_core_news_md")
                self.nlp_spacy = spacy.load("pt_core_news_md")
            except:
                print("AVISO: Não foi possível carregar o modelo spaCy. Using fallback mode.")
                self.nlp_spacy = None
        
        # Padrões para perguntas comuns
        self.learning_patterns = [
            (r'(?:o que|que|quem|qual) (?:é|são) (.*?)[\?]?$', self._handle_what_is),
            (r'como (?:fazer|funciona|posso) (.*?)[\?]?$', self._handle_how_to),
            (r'quem (?:é|são) (.*?)[\?]?$', self._handle_who_is),
            (r'qual (?:é|são) (.*?)[\?]?$', self._handle_which_is),
            (r'onde (?:fica|está|encontro|é) (.*?)[\?]?$', self._handle_where_is),
            (r'por que (.*?)[\?]?$', self._handle_why_is),
            (r'quando (.*?)[\?]?$', self._handle_when_is)
        ]
        
        # Diálogos genéricos para quando não souber responder
        self.dialogue_fillers = [
            "Interessante, mas não aprendi sobre isso ainda. Você pode me ensinar mais?",
            "Hmm, não tenho conhecimento sobre isso. Pode me explicar?",
            "Essa é uma boa pergunta, mas ainda não tenho informações sobre isso na minha base.",
            "Estou aprendendo aos poucos. Poderia me contar mais sobre isso?",
            "Ainda estou desenvolvendo meu conhecimento. O que você sabe sobre isso?"
        ]
        
        # Expressões de saudação
        self.greetings = [
            "olá", "oi", "e aí", "bom dia", "boa tarde", "boa noite", "tudo bem", "como vai"
        ]
        
        # Despedidas
        self.farewells = [
            "tchau", "até logo", "até mais", "adeus", "até a próxima", "até breve"
        ]
        
        # Memória de curto prazo para contexto
        self.conversation_context = {
            "current_topic": None,
            "last_entities": [],
            "follow_up_questions": []
        }

    def process_input(self, user_input):
        """Processa a entrada do usuário e gera uma resposta adequada"""
        # Normaliza e prepara o texto
        user_input = user_input.strip()
        
        # Verificar comandos especiais primeiro
        if user_input.lower().startswith('aprender:'):
            return self._process_learning(user_input[9:].strip())
        
        elif user_input.lower().startswith('calcular:'):
            return self._process_calculation(user_input[10:].strip())
        
        elif user_input.lower().startswith('revisar conhecimento'):
            return self._review_old_knowledge()
        
        elif user_input.lower().startswith('inferir:'):
            return self._infer_relationship(user_input[8:].strip())
        
        elif user_input.lower().startswith('analisar código:'):
            return self._analyze_code(user_input[15:].strip())
        
        # Verificar se é uma saudação ou despedida
        if self._is_greeting(user_input.lower()):
            return self._handle_greeting(user_input)
        
        if self._is_farewell(user_input.lower()):
            return self._handle_farewell(user_input)
        
        # Atualizar contexto da conversa
        self._update_conversation_context(user_input)
        
        # Verificar padrões de perguntas
        for pattern, handler in self.learning_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                return handler(match.group(1))
        
        # Busca semântica na base de conhecimento
        search_results = self.knowledge_base.semantic_search(user_input)
        
        if search_results:
            return self._formulate_knowledgeable_response(search_results, user_input)
        
        # Extração de entidades como último recurso
        entity_info = self._extract_entities(user_input)
        if entity_info:
            self.conversation_context["last_entities"] = entity_info.split("; ")
            return f"Você mencionou: {entity_info}. Infelizmente, ainda não aprendi sobre isso. Você pode me ensinar?"
        
        # Se nada funcionar, use uma resposta padrão
        return self._generate_default_response(user_input)

    def _process_learning(self, learning_input):
        """Processa comandos de aprendizado"""
        # Verifica formato tradicional com '='
        if '=' in learning_input:
            parts = learning_input.split('=', 1)
            topic = parts[0].strip()
            information = parts[1].strip()
        else:
            # Tenta identificar tópico e informação por análise de texto
            parts = learning_input.split(',', 1)
            if len(parts) == 2:
                topic = parts[0].strip()
                information = parts[1].strip()
            else:
                # Última tentativa: dividir em frases
                sentences = nltk.sent_tokenize(learning_input)
                if len(sentences) >= 2:
                    topic = sentences[0].strip()
                    information = ' '.join(sentences[1:]).strip()
                else:
                    return "Para me ensinar, use o formato: 'aprender: tópico = informação' ou me dê um tópico e uma descrição clara."
        
        # Verifica se conseguimos extrair tópico e informação
        if topic and information:
            if self.knowledge_base.add_fact(topic, information):
                responses = [
                    f"Aprendi que {topic} se refere a: {information}. Obrigado!",
                    f"Agora sei sobre '{topic}'. Adicionei isso à minha base de conhecimento.",
                    f"Entendi! '{topic}' está relacionado a '{information}'. Vou lembrar disso."
                ]
                return random.choice(responses)
            else:
                return f"Eu já tinha aprendido sobre '{topic}' com essa informação."
        else:
            return "Não consegui identificar claramente o que você quer me ensinar. Use 'aprender: tópico = informação'."

    def _process_calculation(self, expression):
        """Processa expressões matemáticas"""
        try:
            result = self.calculation_module.evaluate(expression)
            return f"O resultado de {expression} é {result}"
        except Exception as e:
            return f"Não consegui calcular isso. Erro: {str(e)}"

    def _extract_entities(self, text):
        """Extrai entidades nomeadas do texto"""
        if self.nlp_spacy:
            doc = self.nlp_spacy(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            if entities:
                return "; ".join([f"{text}" for text, label in entities])
        
        # Fallback: extração de palavras-chave
        words = nltk.word_tokenize(text)
        words = [w for w in words if w.isalpha() and len(w) > 3]
        return "; ".join(words[:3]) if words else ""

    def _update_conversation_context(self, user_input):
        """Atualiza o contexto da conversa atual"""
        # Identificar tópico principal
        if self.nlp_spacy:
            doc = self.nlp_spacy(user_input)
            nouns = [token.text for token in doc if token.pos_ in ('NOUN', 'PROPN')]
            if nouns:
                self.conversation_context["current_topic"] = nouns[0]
        
        # Gerar possíveis perguntas de acompanhamento
        recent_convs = self.knowledge_base.get_recent_conversations(3)
        combined_text = " ".join([user_input] + [c["user_input"] for c in recent_convs])
        
        # Identificar áreas de interesse para perguntas futuras
        if self.conversation_context["current_topic"]:
            related = self.knowledge_base.infer_relationships(self.conversation_context["current_topic"])
            if related:
                self.conversation_context["follow_up_questions"] = [
                    f"O que você sabe sobre {rel}?" for rel in related[:2]
                ]

    def _is_greeting(self, text):
        """Verifica se o texto é uma saudação"""
        return any(greeting in text for greeting in self.greetings)

    def _is_farewell(self, text):
        """Verifica se o texto é uma despedida"""
        return any(farewell in text for farewell in self.farewells)

    def _handle_greeting(self, text):
        """Responde a uma saudação"""
        time_greetings = {
            "bom dia": "Bom dia! Como posso ajudar você hoje?",
            "boa tarde": "Boa tarde! Em que posso ser útil?",
            "boa noite": "Boa noite! Como posso auxiliar você agora?"
        }
        
        for greeting, response in time_greetings.items():
            if greeting in text.lower():
                return response
        
        responses = [
            "Olá! Como posso ajudar?",
            "Oi! O que você gostaria de saber hoje?",
            "Olá! Em que posso ser útil para você?",
            "Ei, como vai? No que posso ajudar?"
        ]
        return random.choice(responses)

    def _handle_farewell(self, text):
        """Responde a uma despedida"""
        responses = [
            "Até logo! Foi bom conversar com você!",
            "Tchau! Volte quando quiser aprender mais!",
            "Até a próxima! Estou aqui quando precisar.",
            "Adeus! Espero ter ajudado. Volte sempre!"
        ]
        return random.choice(responses)

    def _formulate_knowledgeable_response(self, search_results, query):
        """Formula uma resposta baseada nos resultados da busca de conhecimento"""
        if not search_results:
            return self._generate_default_response(query)
        
        # Extrai tópicos e informações dos resultados
        topics_info = []
        for result in search_results:
            parts = result.split(':', 1)
            if len(parts) == 2:
                topics_info.append((parts[0].strip(), parts[1].strip()))
        
        # Se temos apenas um resultado, formule uma resposta direta
        if len(topics_info) == 1:
            topic, info = topics_info[0]
            templates = [
                f"Sobre {topic}, eu aprendi que: {info}",
                f"Em relação a '{topic}', sei que {info}",
                f"Baseado no que aprendi, {topic} se refere a {info}",
                f"{info} - isso é o que sei sobre {topic}."
            ]
            return random.choice(templates)
        
        # Se temos vários resultados, combine-os
        else:
            response = "Encontrei algumas informações relevantes:\n\n"
            for i, (topic, info) in enumerate(topics_info[:3], 1):
                response += f"{i}. Sobre {topic}: {info}\n"
            
            # Adicionar pergunta de acompanhamento
            if self.conversation_context["follow_up_questions"]:
                follow_up = random.choice(self.conversation_context["follow_up_questions"])
                response += f"\nGostaria de saber mais sobre isso? {follow_up}"
                
            return response.strip()

    def _handle_what_is(self, topic):
        """Responde a perguntas do tipo 'O que é X'"""
        facts = self.knowledge_base.get_facts_about(topic)
        if facts:
            return f"{topic.capitalize()} é {facts[0]}"
        
        # Tentar busca semântica
        semantic_results = self.knowledge_base.semantic_search(f"o que é {topic}")
        if semantic_results:
            return f"Baseado no que aprendi: {semantic_results[0].split(':', 1)[1].strip()}"
        
        return f"Ainda não aprendi o que é '{topic}'. Você pode me ensinar usando 'aprender: {topic} = sua explicação'."

    def _handle_how_to(self, topic):
        """Responde a perguntas do tipo 'Como fazer X'"""
        facts = self.knowledge_base.get_facts_about(f"como {topic}")
        
        if not facts:
            facts = self.knowledge_base.get_facts_about(topic)
            
        if facts:
            return f"Para {topic}, você pode: {facts[0]}"
        
        # Tentar busca semântica
        semantic_results = self.knowledge_base.semantic_search(f"como {topic}")
        if semantic_results:
            return f"Sobre como {topic}: {semantic_results[0].split(':', 1)[1].strip()}"
        
        return f"Ainda não sei como {topic}. Você pode me ensinar usando 'aprender: como {topic} = explicação do processo'."

    def _handle_who_is(self, topic):
        """Responde a perguntas do tipo 'Quem é X'"""
        facts = self.knowledge_base.get_facts_about(topic)
        if facts:
            return f"{topic.capitalize()} é {facts[0]}"
            
        # Tentar busca semântica
        semantic_results = self.knowledge_base.semantic_search(f"quem é {topic}")
        if semantic_results:
            return f"Sobre {topic}: {semantic_results[0].split(':', 1)[1].strip()}"
            
        return f"Ainda não conheço '{topic}'. Você pode me ensinar usando 'aprender: {topic} = informação sobre essa pessoa'."

    def _handle_which_is(self, topic):
        """Responde a perguntas do tipo 'Qual é X'"""
        facts = self.knowledge_base.get_facts_about(f"qual {topic}")
        if not facts:
            facts = self.knowledge_base.get_facts_about(topic)
            
        if facts:
            return f"Sobre qual {topic}: {facts[0]}"
            
        # Tentar busca semântica
        semantic_results = self.knowledge_base.semantic_search(f"qual {topic}")
        if semantic_results:
            return f"Sobre {topic}: {semantic_results[0].split(':', 1)[1].strip()}"
            
        return f"Ainda não sei qual {topic}. Você pode me ensinar usando 'aprender: qual {topic} = resposta'."

    def _handle_where_is(self, topic):
        """Responde a perguntas do tipo 'Onde está X'"""
        facts = self.knowledge_base.get_facts_about(f"onde {topic}")
        if not facts:
            facts = self.knowledge_base.get_facts_about(topic)
            
        if facts:
            return f"Sobre onde {topic}: {facts[0]}"
            
        # Tentar busca semântica
        semantic_results = self.knowledge_base.semantic_search(f"onde {topic}")
        if semantic_results:
            return f"Sobre a localização de {topic}: {semantic_results[0].split(':', 1)[1].strip()}"
            
        return f"Ainda não sei onde {topic}. Você pode me ensinar usando 'aprender: onde {topic} = localização'."

    def _handle_why_is(self, topic):
        """Responde a perguntas do tipo 'Por que X'"""
        facts = self.knowledge_base.get_facts_about(f"por que {topic}")
        if not facts:
            facts = self.knowledge_base.get_facts_about(topic)
            
        if facts:
            return f"Sobre por que {topic}: {facts[0]}"
            
        # Tentar busca semântica
        semantic_results = self.knowledge_base.semantic_search(f"por que {topic}")
        if semantic_results:
            return f"Sobre o motivo de {topic}: {semantic_results[0].split(':', 1)[1].strip()}"
            
        return f"Ainda não sei por que {topic}. Você pode me ensinar usando 'aprender: por que {topic} = explicação'."

    def _handle_when_is(self, topic):
        """Responde a perguntas do tipo 'Quando X'"""
        facts = self.knowledge_base.get_facts_about(f"quando {topic}")
        if not facts:
            facts = self.knowledge_base.get_facts_about(topic)
            
        if facts:
            return f"Sobre quando {topic}: {facts[0]}"
            
        # Tentar busca semântica
        semantic_results = self.knowledge_base.semantic_search(f"quando {topic}")
        if semantic_results:
            return f"Sobre o momento de {topic}: {semantic_results[0].split(':', 1)[1].strip()}"
            
        return f"Ainda não sei quando {topic}. Você pode me ensinar usando 'aprender: quando {topic} = informação temporal'."

    def _generate_default_response(self, user_input):
        """Gera uma resposta padrão quando não há conhecimento específico"""
        # Verificar se a entrada tem palavras conhecidas no vocabulário
        words = nltk.word_tokenize(user_input.lower())
        known_words = [w for w in words if self.knowledge_base.is_known_word(w)]
        
        if known_words:
            return f"Reconheço palavras como '{', '.join(known_words[:3])}', mas não tenho conhecimento completo sobre o que você perguntou. Você pode me ensinar?"
        
        return random.choice(self.dialogue_fillers)

    def _review_old_knowledge(self):
        """Revisa conhecimentos antigos"""
        old_topics = self.knowledge_base.get_old_facts(limit=3)
        if not old_topics:
            return "Todos os conhecimentos estão atualizados!"

        response = "Aqui estão alguns conhecimentos que você pode querer revisar ou atualizar:\n"
        for topic in old_topics:
            facts = self.knowledge_base.get_facts_about(topic)
            if facts:
                response += f"- {topic}: {facts[0]}\n"
        return response.strip()

    def _infer_relationship(self, concept):
        """Infere relacionamentos entre conceitos"""
        related = self.knowledge_base.infer_relationships(concept)
        if related:
            response = f"A partir do que aprendi, posso inferir que '{concept}' está relacionado a: "
            response += ", ".join(related)
            return response
        return f"Ainda não tenho informações suficientes para inferir relações sobre '{concept}'. Você pode me ensinar mais?"

    def _analyze_code(self, code):
        """Analisa código de programação"""
        # Verifica se temos conhecimento sobre programação
        prog_facts = []
        for lang in ['python', 'javascript', 'java', 'c++', 'programação', 'código']:
            facts = self.knowledge_base.get_facts_about(lang)
            prog_facts.extend(facts)
        
        if not prog_facts:
            return "Ainda não aprendi o suficiente sobre programação para analisar código. Você pode me ensinar conceitos de programação primeiro?"
        
        # Identificar linguagem (simples)
        language = "desconhecida"
        if "def " in code or "import " in code or "print(" in code:
            language = "Python"
        elif "function" in code or "var " in code or "let " in code or "const " in code:
            language = "JavaScript"
        elif "public class" in code or "public static void" in code:
            language = "Java"
        elif "#include" in code or "int main" in code:
            language = "C/C++"
        
        # Análise muito básica
        lines = code.strip().split("\n")
        num_lines = len(lines)
        
        return f"Analisei o código em {language} com {num_lines} linhas. Para uma análise mais detalhada, precisaria aprender mais sobre sintaxe e boas práticas de programação."