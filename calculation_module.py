import re
import math
import operator
from decimal import Decimal, InvalidOperation

class CalculationModule:
    def __init__(self):
        # Dicionário com operadores básicos
        self.operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '^': operator.pow,
            '%': operator.mod
        }
        
        # Funções matemáticas disponíveis
        self.functions = {
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log10,
            'ln': math.log,
            'abs': abs,
            'round': round
        }
    
    def evaluate(self, expression):
        """Avalia uma expressão matemática."""
        # Remover espaços em branco
        expression = expression.replace(' ', '')
        
        # Tentar executar a expressão diretamente (mais seguro que eval)
        try:
            # Substituir funções conhecidas por suas implementações
            for func_name, func in self.functions.items():
                pattern = f"{func_name}\(([^)]+)\)"
                while re.search(pattern, expression):
                    match = re.search(pattern, expression)
                    arg_expr = match.group(1)
                    # Avalia o argumento recursivamente
                    arg_value = self.evaluate(arg_expr)
                    # Calcula o resultado da função
                    func_result = func(arg_value)
                    # Substitui a função pelo resultado
                    expression = expression[:match.start()] + str(func_result) + expression[match.end():]
            
            # Converter ^ para ** (potência em Python)
            expression = expression.replace('^', '**')
            
            # Avaliar a expressão final de forma segura
            # Limitamos as variáveis disponíveis
            allowed_names = {"math": math}
            
            # Construir um dicionário de funções seguras
            code = compile(expression, "<string>", "eval")
            for name in code.co_names:
                if name not in allowed_names:
                    raise NameError(f"O uso de '{name}' não é permitido")
            
            return eval(expression, {"__builtins__": {}}, allowed_names)
            
        except Exception as e:
            # Tratar erros comuns
            if "division by zero" in str(e):
                raise ValueError("Divisão por zero não é permitida")
            elif "invalid syntax" in str(e):
                raise ValueError("Sintaxe inválida na expressão")
            else:
                raise ValueError(f"Erro ao calcular: {str(e)}")
    
    def learn_formula(self, name, formula):
        """
        Permite que o sistema aprenda uma nova fórmula.
        Não implementado para manter a segurança, mas poderia ser expandido no futuro.
        """
        # Esta função seria implementada em uma versão mais avançada
        # com análise de segurança robusta
        return "O aprendizado de fórmulas personalizadas não está disponível nesta versão."