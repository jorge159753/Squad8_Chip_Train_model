# Escolha a imagem base com Python
FROM python:3.12.5

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de requisitos (dependências)
COPY requirements.txt .

# Instala as dependências necessárias
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código da aplicação para o container
COPY /app /app

# Comando padrão para rodar o script principal
CMD ["python", "modeltrain.py"]
