FROM python:3.9-slim

WORKDIR /app

# On copie tes fichiers actuels
COPY . .

# On installe ta liste existante (+ joblib)
RUN pip install --no-cache-dir -r requirements.txt

# Le port obligatoire pour Hugging Face
EXPOSE 7860

# ⚠️ Vérifie bien que le nom du fichier ci-dessous est EXACTEMENT celui de ton fichier principal
# Si tu l'as renommé "1_Dashboard.py", laisse comme ça. Sinon mets le vrai nom.
CMD ["streamlit", "run", "1_Dashboard.py", "--server.port", "7860", "--server.address", "0.0.0.0"]