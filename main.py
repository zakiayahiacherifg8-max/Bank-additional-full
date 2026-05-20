import pandas as pd

try:
    # On précise bien le chemin vers le dossier data
    df = pd.read_csv('data/bank-additional-full.csv', sep=';')
    print("✅ Données chargées avec succès !")
    print(f"Lignes : {df.shape[0]}, Colonnes : {df.shape[1]}")
    print("\n--- Top 5 des métiers dans la base ---")
    print(df['job'].value_counts().head())
except Exception as e:
    print(f"❌ Erreur : {e}")