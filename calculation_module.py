import math

class CalculationModule:
    def __init__(self):
        # Inicializar operadores permitidos
        self.allowed_operators = {"+", "-", "*", "/", "^", "%"}
        self.functions = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sqrt": math.sqrt,
            "log": math.log,
            "exp": math.exp,
            "abs": abs
        }

    def evaluate(self, expression):
        """Avalia uma expressão matemática de forma segura."""
        try:
            # Substituir operadores para segurança
            sanitized_expression = self.sanitize_expression(expression)
            
            # Avaliar a expressão
            result = eval(sanitized_expression, {"__builtins__": None}, self.functions)
            return result
        except ZeroDivisionError:
            return "Erro: Divisão por zero."
        except Exception as e:
            return f"Erro ao calcular: {str(e)}"

    def sanitize_expression(self, expression):
        """Sanitiza a expressão antes de avaliar."""
        sanitized = ""
        for char in expression:
            if char.isdigit() or char in self.allowed_operators or char.isalpha() or char == ".":
                sanitized += char
            else:
                raise ValueError(f"Caractere não permitido na expressão: {char}")
        
        # Verificar funções matemáticas na expressão
        for func in self.functions.keys():
            sanitized = sanitized.replace(func, f"self.functions['{func}']")
        
        return sanitized

    def calculate_with_variables(self, expression, variables):
        """Permite calcular expressões que incluem variáveis."""
        try:
            sanitized_expression = self.sanitize_expression(expression)
            # Substituir variáveis na expressão
            for var, value in variables.items():
                sanitized_expression = sanitized_expression.replace(var, str(value))
            
            # Avaliar a expressão
            result = eval(sanitized_expression, {"__builtins__": None}, self.functions)
            return result
        except Exception as e:
            return f"Erro ao calcular com variáveis: {str(e)}"