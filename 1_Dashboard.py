import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import boto3
import os
from io import StringIO
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
load_dotenv()

st.set_page_config(
    page_title="Bank Marketing Insights",
    page_icon="🏦",
    layout="wide"
)

# --- 2. FONCTIONS BACKEND (S3) ---
@st.cache_data
def charger_data_s3(nom_du_fichier):
    # Connexion S3
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('ACCESS_KEY'),
        aws_secret_access_key=os.getenv('SECRET_KEY'),
        region_name="eu-west-3"
    )
    reponse = s3_client.get_object(Bucket=os.getenv('BUCKET_NAME'), Key=nom_du_fichier)
    contenu = reponse['Body'].read().decode('utf-8')
    
    # Lecture avec le bon séparateur (point-virgule)
    return pd.read_csv(StringIO(contenu), sep=';')

# --- 3. CHARGEMENT ET CALCULS ---
try:
    with st.spinner('Chargement des données...'):
        df = charger_data_s3("bank_marketing_cleaned_v1.csv")
        
        # Définition de la cible
        COLONNE_CIBLE = 'souscription'
        
        # Création d'une colonne numérique pour les calculs (1=Yes, 0=No)
        df['target_num'] = df[COLONNE_CIBLE].apply(lambda x: 1 if x == 'yes' else 0)
        
        # Calcul du taux de conversion global
        conversion_rate = (df[COLONNE_CIBLE].value_counts(normalize=True).get('yes', 0)) * 100

except Exception as e:
    st.error(f"Erreur technique : {e}")
    st.stop()

# --- 4. INTERFACE UTILISATEUR ---

# Sidebar simple
st.sidebar.title("Navigation")
st.sidebar.success("Données connectées S3")
st.sidebar.markdown("---")
st.sidebar.info("Projet Bank Marketing")

# Titre Principal
st.title("🏦 Optimisation des Campagnes Marketing")
st.markdown("### Analyse de la performance et ciblage prédictif")
st.markdown("---")

# --- SECTION KPI MACRO ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Volume Clients", value=f"{df.shape[0]:,}".replace(",", " "))
with col2:
    st.metric(label="Taux de Conversion", value=f"{conversion_rate:.2f} %")
with col3:
    st.metric(label="Âge Moyen", value=f"{df['age'].mean():.0f} ans")
with col4:
    st.metric(label="Durée Moyenne", value=f"{df['duration'].mean()/60:.1f} min")

st.markdown("---")

# --- SECTION 1 : APERÇU DES DONNÉES ---
with st.expander("👁️ Afficher un aperçu des données brutes"):
    st.dataframe(df.head())

# --- SECTION 2 : LE PROBLÈME (Déséquilibre) ---
st.header("1. Analyse de la Cible (Target)")

c1, c2 = st.columns([1, 1])

with c1:
    # GRAPHIQUE 1 : Distribution Globale
    counts = df[COLONNE_CIBLE].value_counts().reset_index()
    counts.columns = ['Résultat', 'Nombre']
    
    fig = px.bar(
        counts, 
        x='Résultat', 
        y='Nombre', 
        color='Résultat',
        text_auto=True,
        color_discrete_map={'no': "#CA2103", 'yes': "#05662A"}, 
        title="Distribution des Souscriptions (Oui/Non)"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.warning("⚠️ **ALERTE DATA : DÉSÉQUILIBRE**")
    st.markdown("""
    On constate une majorité écrasante de refus (**'no'**).
    
    **Impact pour le Machine Learning :**
    * L'Accuracy (Précision globale) sera trompeuse.
    * Un modèle qui prédit "Non" tout le temps aura ~88% de réussite.
    
    👉 **Action :** 1.  **Stratifier** nos échantillons.
    2.  Juger le modèle sur son **Recall** (ne rater aucune vente).
    """)

# BLOC ZOOM (LE SLICER EST ICI)
with c2:
    st.markdown("#### 🔍 Zoom sur la répartition")
    
    critere = st.selectbox(
        "Voir le détail par :",
        ["Métier", "Statut Matrimonial", "Niveau d'Études"]
    )
    
    col_map = {
        "Métier": "metier",
        "Statut Matrimonial": "statut_matrimonial",
        "Niveau d'Études": "niveau_etudes" 
    }
    col_choisie = col_map[critere]
    
    if col_choisie in df.columns:
        df_zoom = df.groupby([col_choisie, COLONNE_CIBLE]).size().reset_index(name='Nombre')
        
        fig_zoom = px.bar(
            df_zoom,
            x='Nombre',
            y=col_choisie,
            color=COLONNE_CIBLE,
            orientation='h',
            title=f"Répartition {critere} x Souscription",
            color_discrete_map={'no': "#CA2103", 'yes': "#05662A"},
            barmode='stack'
        )
        fig_zoom.update_layout(legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_zoom, use_container_width=True)
    else:
        st.info("Donnée non disponible.")
    
st.markdown("---")

# --- SECTION 3 : PROFIL CLIENT (WHO) ---
st.header("2. PROFILING : QUI EST LE CLIENT IDÉAL ?")

st.markdown("Comparaison **Volume** (qui on appelle) vs **Performance** (qui signe).")

# 1. PRÉPARATION DES DONNÉES
df_job = df.groupby('metier').agg(
    Volume=('souscription', 'count'),
    Conversion_Rate=('target_num', 'mean')
).reset_index()
# Conversion en % et Arrondi à 2 décimales
df_job['Conversion_Rate'] = (df_job['Conversion_Rate'] * 100).round(2)
df_job = df_job.sort_values(by='Conversion_Rate', ascending=False)

# 2. VISUALISATION MÉTIER
st.subheader("A. Analyse par Métier (Job)")

col_job1, col_job2 = st.columns(2)

with col_job1:
    # Volume
    df_vol = df_job.sort_values(by='Volume', ascending=True)
    fig_vol = px.bar(
        df_vol, 
        x='Volume', 
        y='metier', 
        orientation='h',
        title="Volume d'appels par métier",
        text_auto=True,
        color_discrete_sequence=["#E6A66A"]
    )
    st.plotly_chart(fig_vol, use_container_width=True)

with col_job2:
    # Performance
    fig_perf = px.bar(
        df_job, 
        x='Conversion_Rate', 
        y='metier',
        orientation='h',
        title="Taux de Conversion (%)",
        text_auto='.2f', # Formatage affichage 2 décimales
        color='Conversion_Rate',
        color_continuous_scale=["#EED9C5","#E6A66A","#F18C2D","#884506"]
    )
    fig_perf.add_vline(x=conversion_rate, line_dash="dash", line_color="Brown", annotation_text="Moyenne")
    st.plotly_chart(fig_perf, use_container_width=True)

# Commentaire Business
top_job = df_job.iloc[0]['metier']
top_perf = df_job.iloc[0]['Conversion_Rate']
flop_job = df_job.iloc[-1]['metier']

st.info(f"""
**Observation Business :** Le métier **{top_job}** est le plus performant avec **{top_perf:.2f}%** de réussite.
A l'inverse, **{flop_job}** convertit mal, malgré un volume souvent élevé.

👉 *Stratégie : Réallouer les efforts des profils à faible rendement vers les profils performants.*
""")


# 3. VISUALISATION AGE & STATUT
st.subheader("B. Analyse Démographique")

col_age1, col_age2 = st.columns(2)

with col_age1:
    if 'age_group' in df.columns:
        df_age = df.groupby('age_group')['target_num'].mean().reset_index()
        # Arrondi
        df_age['target_num'] = (df_age['target_num'] * 100).round(2)
        
        fig_age = px.bar(
            df_age, 
            x='age_group', 
            y='target_num',
            title="Performance par Tranche d'Âge",
            text_auto='.2f',
            color='target_num',
            color_continuous_scale=["#FFEEE5","#DB9452","#D8802E","#FF7B00"],
            labels={'target_num': 'Conversion (%)'}
        )
        st.plotly_chart(fig_age, use_container_width=True)
    else:
        st.warning("Colonne 'age_group' introuvable.")

with col_age2:
    df_statut = df.groupby('statut_matrimonial')['target_num'].mean().reset_index()
    # Arrondi
    df_statut['target_num'] = (df_statut['target_num'] * 100).round(2)
    
    fig_statut = px.bar(
        df_statut, 
        x='statut_matrimonial', 
        y='target_num',
        title="Performance par Statut",
        text_auto='.2f',
        color='target_num',
        color_continuous_scale='Oranges',
        labels={'target_num': 'Conversion (%)'}
    )
    st.plotly_chart(fig_statut, use_container_width=True)

# Calculs auto pour texte
top_age_group = df.groupby('age_group')['target_num'].mean().idxmax() if 'age_group' in df.columns else "N/A"
perf_age = df.groupby('age_group')['target_num'].mean().max() * 100 if 'age_group' in df.columns else 0
top_statut = df.groupby('statut_matrimonial')['target_num'].mean().idxmax()
perf_statut = df.groupby('statut_matrimonial')['target_num'].mean().max() * 100

st.info(f"""
**Observation Business :** Sur le plan démographique, deux signaux forts se dégagent :
1.  **L'Âge :** Le segment **{top_age_group}** est le plus réactif avec **{perf_age:.2f}%** de conversion.
2.  **La Situation :** Les profils **{top_statut}** (statut matrimonial) surperforment avec **{perf_statut:.2f}%** de réussite.

👉 *Stratégie : Ne vendez pas le même produit à tout le monde. Adaptez le discours.*
""")

st.markdown("---")

# --- SECTION 4 : STRATÉGIE TEMPORELLE (WHEN) ---
st.header("3. TIMING : QUAND LANCER LES CAMPAGNES ?")

st.markdown("Analyse de la **Saisonnalité** (Mois) et de la **Pression Marketing**.")

# 1. ANALYSE MENSUELLE (COMBO CHART AVEC PLOTLY GO)
st.subheader("A. Le Paradoxe du Mois de Mai")

ordre_mois = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
df_mois = df.groupby('mois').agg(
    Volume=('souscription', 'count'),
    Taux_Conversion=('target_num', 'mean')
).reindex(ordre_mois).dropna().reset_index()
# Arrondi
df_mois['Taux_Conversion'] = (df_mois['Taux_Conversion'] * 100).round(2)

col_mois1, col_mois2 = st.columns([2, 1])

with col_mois1:
    fig_combo = go.Figure()

    # Barres (Volume)
    fig_combo.add_trace(go.Bar(
        x=df_mois['mois'],
        y=df_mois['Volume'],
        name='Volume Appels',
        marker_color='lightgrey'
    ))

    # Ligne (Taux)
    fig_combo.add_trace(go.Scatter(
        x=df_mois['mois'],
        y=df_mois['Taux_Conversion'],
        name='Taux de Réussite (%)',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='red', width=3),
        hovertemplate='%{y:.2f}%' # Template survol propre
    ))

    # Layout double axe
    fig_combo.update_layout(
        title="Volume vs Performance par Mois",
        yaxis=dict(title="Volume d'appels"),
        yaxis2=dict(
            title="Taux de Conversion (%)",
            overlaying='y',
            side='right',
            range=[0, df_mois['Taux_Conversion'].max()*1.2]
        ),
        legend=dict(x=0, y=1.1, orientation='h'),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig_combo, use_container_width=True)

    # TEXTE RICHE D'ORIGINE
    st.info("""
    📉 **Analyse :**
    Le mois de **Mai (may)** : C'est le pic d'appels, mais le taux de réussite s'effondre.
    
    ✅ **Opportunité :**
    Les mois de **Mars, Septembre, Octobre** ont moins d'appels mais d'excellents taux de conversion.
    """)

# ZOOM MOIS (DROITE)
with col_mois2:
    st.markdown("#### 🔍 Zoom : Qui a-t-on appelé ?")
    
    mois_select = st.selectbox("Sélectionnez un mois :", ordre_mois, index=4) # index 4 = may
    
    df_zoom_month = df[df['mois'] == mois_select]
    df_zoom_job = df_zoom_month['metier'].value_counts().reset_index()
    df_zoom_job.columns = ['Metier', 'Volume']
    
    fig_zoom_month = px.bar(
        df_zoom_job.sort_values('Volume'), 
        x='Volume', 
        y='Metier', 
        orientation='h',
        title=f"Répartition en {mois_select.upper()}",
        text_auto=True,
        color_discrete_sequence=["#E6A66A"]
    )
    fig_zoom_month.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
    st.plotly_chart(fig_zoom_month, use_container_width=True)


# 2. ANALYSE DE LA PRESSION
st.subheader("B. Acharnement vs Efficacité")

df_campaign = df.groupby('campaign')['target_num'].mean().reset_index()
# Arrondi
df_campaign['target_num'] = (df_campaign['target_num'] * 100).round(2)
df_campaign = df_campaign[df_campaign['campaign'] <= 10]

col_cam1, col_cam2 = st.columns([2, 1])

with col_cam1:
    fig_press = px.line(
        df_campaign,
        x='campaign',
        y='target_num',
        markers=True,
        title="Chute de la conversion après X appels",
        labels={'target_num': 'Succès (%)', 'campaign': 'Nb contacts'}
    )
    fig_press.add_vline(x=3, line_dash="dash", line_color="red", annotation_text="Zone Harcèlement")
    # Mise à jour du format de survol
    fig_press.update_traces(hovertemplate='Appels: %{x}<br>Succès: %{y:.2f}%')
    
    st.plotly_chart(fig_press, use_container_width=True)

with col_cam2:
    st.warning("""
    ⚠️ **Stop ou Encore ?**
    La courbe montre clairement qu'après **3 appels**, la probabilité de vente devient quasi-nulle.
    Continuer à appeler au-delà de 3 fois coûte de l'argent et risque "faire fuir" le client.
    """)

st.markdown("---")

# --- SECTION 5 : ANALYSE CROISÉE (THE SNIPER VIEW) ---
st.header("4. CIBLAGE CHIRURGICAL : QUI & QUAND ?")

st.markdown("Carte de chaleur (Heatmap) croisée **Métier x Mois**.")

# PRÉPARATION PIVOT
pivot_table = df.pivot_table(
    values='target_num',
    index='metier',
    columns='mois',
    aggfunc='mean'
)
# Arrondi du pivot direct
pivot_table = (pivot_table * 100).round(2)
pivot_table = pivot_table.reindex(columns=ordre_mois)

# HEATMAP PLOTLY
fig_heat = px.imshow(
    pivot_table,
    labels=dict(x="Mois", y="Métier", color="Conversion (%)"),
    x=pivot_table.columns,
    y=pivot_table.index,
    color_continuous_scale='RdYlGn',
    text_auto=".2f", # 2 décimales dans les cases
    aspect="auto"
)
fig_heat.update_layout(title="Matrice de Rentabilité")
st.plotly_chart(fig_heat, use_container_width=True)

# --- INSIGHTS SPÉCIFIQUES ---
st.markdown("### 💡 Analyse détaillée de la Matrice")

# TEXTES COMPLETS (RESTAURÉS)
col_alerte, col_opportunite = st.columns(2)

with col_alerte:
    st.warning("""
    ### ⚠️ ALERTE QUALITÉ (Data Quality)
    **Le mystère "Unknown" en Avril (85.7% de réussite) :**
    
    Nous observons un taux de conversion record chez les clients dont le métier est inconnu (`unknown`) en Avril.
    
    👉 **Le Problème :** C'est une perte d'information critique ! Les commerciaux ont vendu, mais ils n'ont pas rempli le CRM.
    **Action :** Rappeler aux équipes l'importance de qualifier la fiche client (le champ métier est obligatoire).
    """)

with col_opportunite:
    st.success("""
    ### 🚀 OPPORTUNITÉ DE MARCHÉ
    **Le "Carton Plein" des Entrepreneurs en Mars (100%)**
    
    Les entrepreneurs convertissent à **100%** sur le mois de Mars.
    
    👉 **L'Explication Business :**
    * **Fiscalité :** Fin de l'exercice fiscal et ouverture des nouveaux budgets.
    * **Écosystème :** Saison des **Salons Professionnels** et des **Concours** (Recherche de financement).
    
    **Stratégie :** Lancer une campagne "Crédit Pro" spécifique fin Février.
    """)

# --- CONCLUSION FINALE ---
st.markdown("---")
st.header("🎓 RECOMMANDATIONS STRATÉGIQUES")

# TEXTES COMPLETS (RESTAURÉS)
col_rec1, col_rec2 = st.columns(2)

with col_rec1:
    st.success("""
    ### ✅ CE QU'IL FAUT FAIRE (TOP ACTIONS)
    1.  **Miser sur les Entrepreneurs en Mars 🚀 :** C'est le "Golden Month" (Clôture fiscale & Salons pro). À prioriser absolument.
    2.  **Cibler les extrêmes générationnels :** Les **Étudiants** (Rentrée Mars/Sept) et les **Retraités** (Placement en Oct/Déc) sont les plus rentables.
    3.  **Respecter la règle de 3 :** Si le client ne signe pas au **3ème appel**, abandonner. L'acharnement coûte cher et rapporte peu.
    """)

with col_rec2:
    st.error("""
    ### ⛔ CE QU'IL FAUT ÉVITER (PIÈGES)
    1.  **Le "Mirage" du mois de Mai :** C'est le mois avec le plus gros volume d'appels mais le pire taux de réussite. Réduire la pression sur cette période.
    2.  **L'inconnue du CRM (Data Quality) :** Les profils "Unknown" convertissent fort en Avril, mais c'est une anomalie. **Forcer les commerciaux à remplir le champ métier.**
    3.  **L'illusion de l'Accuracy :** Ne pas se fier à la précision globale du futur modèle (88%). Il faudra optimiser le **Recall** (ne rater aucune vente).
    """)

# Signature
st.markdown("---")
st.caption("Dashboard réalisé avec Streamlit & AWS S3 • Données Bank Marketing")