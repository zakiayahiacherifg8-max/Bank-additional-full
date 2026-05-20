import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import boto3
import os
from io import BytesIO
from dotenv import load_dotenv

# --- 1. CONFIGURATION DE LA PAGE ---
load_dotenv()

st.set_page_config(
    page_title="Simulateur Prédictif",
    page_icon="🔮",
    layout="centered"
)

# --- 2. CHARGEMENT DU MODÈLE S3 ---
@st.cache_resource(show_spinner="Réveil de l'IA...")
def charger_modele_s3():
    bucket_name = os.getenv('BUCKET_NAME')
    model_key = "model_bank_marketing_v1.joblib"
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('ACCESS_KEY'),
            aws_secret_access_key=os.getenv('SECRET_KEY'),
            region_name="eu-west-3"
        )
        response = s3.get_object(Bucket=bucket_name, Key=model_key)
        model_bytes = BytesIO(response['Body'].read())
        return joblib.load(model_bytes)
    except Exception as e:
        st.error(f"❌ Erreur S3 : {e}")
        return None

model = charger_modele_s3()

# --- 3. SIDEBAR : INTERFACE (UX FUSIONNÉE) ---
st.sidebar.header("🎯 Leviers Prioritaires")

resultat_precedent = st.sidebar.selectbox("Résultat campagne précédente", ['no existant', 'failure', 'success'])
pret_immo = st.sidebar.selectbox("A déjà un Prêt Immobilier ?", ['no', 'yes'])
age = st.sidebar.selectbox("Âge du client", options=list(range(18, 96)), index=17) 
solde_bancaire = st.sidebar.number_input("Solde Bancaire (€)", -5000, 100000, 1500)
previous = st.sidebar.slider("Nombre d'interactions passées", 0, 30, 0)

with st.sidebar.expander("⚙️ Paramètres Avancés"):
    segment_contact = st.selectbox("Segment Contact", ['Jamais contacte', 'Ancien (>90j)', 'Intermediaire (31-90j)', 'Recent (0-30j)'])
    metier = st.selectbox("Métier", ['management', 'technician', 'entrepreneur', 'blue-collar', 'unknown', 'retired', 'admin.', 'services', 'self-employed', 'unemployed', 'housemaid', 'student'])
    statut_matrimonial = st.selectbox("Statut Matrimonial", ['married', 'single', 'divorced'])
    niveau_etudes = st.selectbox("Niveau d'Études", ['tertiary', 'secondary', 'unknown', 'primary'])
    mois = st.selectbox("Mois de l'appel", ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'], index=4)
    day = st.slider("Jour du mois", 1, 31, 15)
    campaign = st.slider("Nb appels cette campagne", 1, 10, 1)
    defaut_credit = st.selectbox("Défaut Crédit", ['no','yes'])
    pret_conso = st.selectbox("Prêt Conso", ['no','yes'])

# --- 4. LOGIQUE DE PRÉDICTION ---
if st.sidebar.button("🎯 Lancer la prédiction"):
    input_data = pd.DataFrame([{
        'age': age,
        'solde_bancaire': solde_bancaire,
        'day': day,
        'campaign': campaign,
        'pdays': -1,
        'previous': previous,
        'defaut_credit': defaut_credit,
        'pret_immo': pret_immo,
        'pret_conso': pret_conso,
        'metier': metier,
        'statut_matrimonial': statut_matrimonial,
        'niveau_etudes': niveau_etudes,
        'mois': mois,
        'resultat_precedent': resultat_precedent,
        'segment_contact': segment_contact
    }])

    cat_cols = ['metier','statut_matrimonial','niveau_etudes','defaut_credit', 
                'pret_immo', 'pret_conso', 'mois','resultat_precedent','segment_contact']
    input_data_encoded = pd.get_dummies(input_data, columns=cat_cols)

    model_columns = [
        'age', 'solde_bancaire', 'day', 'campaign', 'pdays', 'previous',
        'defaut_credit_yes', 'pret_immo_yes', 'pret_conso_yes',
        'metier_blue-collar', 'metier_entrepreneur', 'metier_housemaid', 'metier_management', 'metier_retired', 
        'metier_self-employed', 'metier_services', 'metier_student', 'metier_technician', 'metier_unemployed', 'metier_unknown',
        'statut_matrimonial_married', 'statut_matrimonial_single',
        'niveau_etudes_secondary', 'niveau_etudes_tertiary', 'niveau_etudes_unknown',
        'mois_aug', 'mois_dec', 'mois_feb', 'mois_jan', 'mois_jul', 'mois_jun', 'mois_mar', 'mois_may', 'mois_nov', 'mois_oct', 'mois_sep',
        'resultat_precedent_no existant', 'resultat_precedent_success',
        'segment_contact_Intermediaire (31-90j)', 'segment_contact_Jamais contacte', 'segment_contact_Recent (0-30j)'
    ]

    for col in model_columns:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0
    input_data_encoded = input_data_encoded[model_columns]

    proba = model.predict_proba(input_data_encoded)[0][1]
    score = round(proba * 100, 2)

    # --- 5. AFFICHAGE ET RECOMMANDATIONS ---
    st.markdown("---")
    st.markdown(f"### Résultat de l'Analyse IA")
    
    if score >= 40:
        st.success(f"**Score de Propension : {score}% (Potentiel Élevé)**")
    elif score >= 15:
        st.warning(f"**Score de Propension : {score}% (Potentiel Modéré)**")
    else:
        st.info(f"**Score de Propension : {score}% (Potentiel Faible)**")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 15], 'color': "#FF4B4B"},
                {'range': [15, 40], 'color': "#FFAA00"},
                {'range': [40, 100], 'color': "#00BB44"}
            ],
            'bar': {'color': "black"}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🚦 Recommandation ")
    if score >= 40:
        st.success("🟢 **PRIORITÉ HAUTE** : Opportunité immédiate. Client très réceptif. Conclure rapidement en mettant en avant les avantages de l'épargne et la sécurité.")
    elif score >= 15:
        st.warning("🟠 **PRIORITÉ MOYENNE** : Client à potentiel, renforcer l'argumentaire. Le client est hésitant mais captable avec une offre personnalisée axée sur la flexibilité.")
    else:
        st.error("🔴 **PRIORITÉ BASSE** : Ne pas abandonner, mais allouer peu de ressources. Allouer le temps commercial sur des profils plus qualifiés pour maximiser le ROI.")

    # --- CONSEILS COMMERCIAUX (DÉSORMAIS BIEN INDENTÉS) ---
    st.markdown("## 💼 Conseils pour le commercial")

    if score < 30:
        st.info("📉 Faible probabilité de souscription")
        st.markdown("""
        - Ne pas investir trop de temps sur ce client pour le moment  
        - Prévoir un suivi léger dans quelques semaines  
        - Noter les préférences du client pour un futur contact  
        - Rester poli et courtois, maintenir la relation
        """)
    elif score <= 60:
        st.info("⚖️ Probabilité moyenne de souscription")
        st.markdown("""
        - Contacter le client avec un argumentaire personnalisé  
        - Mettre en avant les avantages concrets du produit  
        - Prévoir un suivi rapproché pour répondre aux questions  
        - Identifier les objections possibles et préparer des réponses
        """)
    else:
        st.info("🚀 Forte probabilité de souscription")
        st.markdown("""
        - Priorité haute : contacter rapidement le client  
        - Finaliser la souscription dès que possible  
        - Proposer des services complémentaires adaptés  
        - Insister sur les promotions ou offres exclusives  
        - Confirmer les informations et simplifier le processus
        """)
        