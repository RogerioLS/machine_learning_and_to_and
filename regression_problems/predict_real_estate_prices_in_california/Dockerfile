# Use a base image Python
FROM python:3.9-slim

#Set the working directory in the container
WORKDIR /app

# Copie o arquivo de requisitos e instale as dependências
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie os diretórios do projeto para o container
COPY streamlit /app/streamlit
COPY model_trained /app/model_trained
COPY software_engineering_data_science /app/software_engineering_data_science

# Defina o comando para iniciar o Streamlit
CMD ["streamlit", "run", "/app/streamlit/app.py"]