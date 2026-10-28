"""Dashboard pédagogique : exploration Auto MPG et prédiction Random Forest."""

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1. Configuration et variables utiles
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
NUMERIC_FEATURES = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight_kg",
    "acceleration",
    "model_year",
]
CATEGORICAL_FEATURES = ["origin"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "consumption_l_100km"
COLORS = {"usa": "#2563EB", "japan": "#93C5FD", "europe": "#EF4444"}

st.set_page_config(page_title="Auto MPG — Dashboard et ML", page_icon="🚗", layout="wide")


# 2. Charger et préparer les données (une seule fois grâce au cache)
@st.cache_data(show_spinner=False)
def load_data():
    data = pd.read_csv(DATA_URL)
    data["weight_kg"] = data["weight"] * 0.45359237
    data[TARGET] = 235.214583 / data["mpg"]
    return data


# 3. Construire et évaluer un modèle de régression
@st.cache_resource(show_spinner="Entraînement du Random Forest...")
def train_model(data):
    # Entraîner sur les données complètes, et non sur la sélection du dashboard.
    clean = data.dropna(subset=[TARGET]).copy()
    X = clean[FEATURES]
    y = clean[TARGET]

    # Le jeu de test reste inconnu du modèle pendant l'apprentissage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Valeurs numériques absentes => médiane calculée uniquement sur X_train.
    # Catégorie 'origin' => représentation numérique one-hot.
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("origin", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "forest",
                RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=2,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evaluation = pd.DataFrame(
        {"Réel (L/100 km)": y_test.to_numpy(), "Prédit (L/100 km)": y_pred}
    )
    return model, mae, r2, evaluation


# 4. Charger les données et afficher le titre
st.title("🚗 Exploration des données automobiles")
st.caption("Dataset Auto MPG (1970–1982) — visualisation et apprentissage machine")
st.info(
    "MPG élevé = voiture plus économe. "
    "L/100 km faible = voiture plus économe. "
    "Les prédictions sont des estimations sur des voitures historiques."
)

try:
    df = load_data()
except Exception as error:
    st.error("Impossible de charger le CSV. Vérifie ta connexion Internet.")
    st.exception(error)
    st.stop()


# 5. Barre latérale : filtres d'exploration
st.sidebar.header("Filtres")
origins = ["Toutes"] + sorted(df["origin"].dropna().unique().tolist())
selected_origin = st.sidebar.selectbox("Origine", origins)

all_cylinders = sorted(df["cylinders"].dropna().unique().tolist())
selected_cylinders = st.sidebar.multiselect(
    "Nombre de cylindres", all_cylinders, default=all_cylinders
)

min_year, max_year = int(df["model_year"].min()), int(df["model_year"].max())
selected_years = st.sidebar.slider(
    "Année du modèle", min_year, max_year, (min_year, max_year)
)

min_mpg = st.sidebar.slider(
    "Efficacité minimale (mpg)",
    min_value=float(df["mpg"].min()),
    max_value=float(df["mpg"].max()),
    value=float(df["mpg"].min()),
    step=0.5,
)

filtered = df[
    df["cylinders"].isin(selected_cylinders)
    & df["model_year"].between(*selected_years)
    & (df["mpg"] >= min_mpg)
].copy()
if selected_origin != "Toutes":
    filtered = filtered[filtered["origin"] == selected_origin]


# 6. Tableau de bord descriptif
st.header("1 — Indicateurs et données filtrées")
st.write(f"**{len(filtered)} observations sur {len(df)}**")

if filtered.empty:
    st.warning("Aucun véhicule ne correspond aux filtres. Change les critères à gauche.")
else:
    m1, m2, m3 = st.columns(3)
    m1.metric("Efficacité moyenne", f"{filtered['mpg'].mean():.1f} mpg")
    m2.metric("Consommation moyenne", f"{filtered[TARGET].mean():.1f} L/100 km")
    m3.metric("Puissance moyenne", f"{filtered['horsepower'].mean():.1f} hp")

    st.caption(
        "La consommation moyenne est calculée après conversion de chaque véhicule, "
        "et non en convertissant la moyenne des MPG."
    )
    display_columns = [
        "name", "origin", "model_year", "cylinders", "mpg", TARGET,
        "horsepower", "weight_kg", "acceleration",
    ]
    st.dataframe(filtered[display_columns].round(2), width="stretch", height=270)

    st.header("2 — Choisir un véhicule")
    selected_index = st.selectbox(
        "Véhicule",
        options=filtered.index.tolist(),
        format_func=lambda i: (
            f"{filtered.loc[i, 'name']} — {int(filtered.loc[i, 'model_year'])} "
            f"(ligne {i})"
        ),
    )
    st.dataframe(filtered.loc[[selected_index], display_columns].round(2), width="stretch")

    st.header("3 — Comparaisons graphiques")
    scatter = px.scatter(
        filtered,
        x="weight_kg", y=TARGET, color="origin", hover_name="name",
        color_discrete_map=COLORS,
        title="Poids et consommation : relation entre les deux variables",
        labels={"weight_kg": "Poids (kg)", TARGET: "Consommation (L/100 km)", "origin": "Origine"},
    )
    st.plotly_chart(scatter, width="stretch")

    yearly = filtered.groupby("model_year", as_index=False)["mpg"].mean()
    line = px.line(
        yearly, x="model_year", y="mpg", markers=True,
        title="Efficacité moyenne par année de modèle",
        labels={"model_year": "Année du modèle", "mpg": "Efficacité moyenne (mpg)"},
    )
    st.plotly_chart(line, width="stretch")

    counts = filtered["origin"].value_counts().rename_axis("origin").reset_index(name="nombre")
    bars = px.bar(
        counts, x="origin", y="nombre", color="origin",
        color_discrete_map=COLORS, title="Nombre d'observations par origine",
        labels={"origin": "Origine", "nombre": "Nombre"},
    )
    st.plotly_chart(bars, width="stretch")


# 7. Apprentissage automatique : modèle indépendant des filtres de visualisation
st.header("4 — Prédiction avec Random Forest Regressor")
st.write(
    "Le modèle apprend sur **80 % des véhicules** et s'évalue sur les **20 % restants**. "
    "Les filtres du dashboard ne changent pas son entraînement."
)

model, mae, r2, evaluation = train_model(df)
metric1, metric2 = st.columns(2)
metric1.metric("Erreur absolue moyenne (MAE)", f"{mae:.2f} L/100 km")
metric2.metric("Score R² sur le jeu de test", f"{r2:.3f}")
st.caption("MAE : plus petit = mieux. R² : plus proche de 1 = mieux (il peut être négatif).")

with st.form("prediction_form"):
    st.subheader("Saisis les caractéristiques d'une voiture")
    left, right = st.columns(2)
    with left:
        cylinders = st.selectbox("Nombre de cylindres", all_cylinders, index=1)
        displacement = st.number_input("Cylindrée (cubic inches)", min_value=50.0, max_value=500.0, value=150.0, step=5.0)
        horsepower = st.number_input("Puissance (hp)", min_value=30.0, max_value=250.0, value=100.0, step=5.0)
        origin = st.selectbox("Origine", ["usa", "europe", "japan"])
    with right:
        weight_kg = st.number_input("Poids (kg)", min_value=500.0, max_value=2500.0, value=1200.0, step=25.0)
        acceleration = st.number_input("Accélération (secondes)", min_value=8.0, max_value=25.0, value=15.0, step=0.5)
        model_year = st.slider("Année du modèle", min_year, max_year, 1976)
    submitted = st.form_submit_button("Prédire la consommation", type="primary")

if submitted:
    new_car = pd.DataFrame([{
        "cylinders": cylinders,
        "displacement": displacement,
        "horsepower": horsepower,
        "weight_kg": weight_kg,
        "acceleration": acceleration,
        "model_year": model_year,
        "origin": origin,
    }])
    estimate = float(model.predict(new_car)[0])
    st.success(f"Consommation estimée : **{estimate:.2f} L/100 km**")
    st.write(f"Équivalent : **{235.214583 / estimate:.1f} mpg**")
    st.caption("Estimation statistique, pas une mesure réelle ni une prévision fiable pour une voiture moderne.")

with st.expander("Voir les prédictions sur le jeu de test"):
    st.dataframe(evaluation.round(2), width="stretch")
    validation = px.scatter(
        evaluation, x="Réel (L/100 km)", y="Prédit (L/100 km)",
        title="Valeurs réelles et valeurs prédites",
    )
    low = min(evaluation.min().min(), evaluation.max().max())
    high = max(evaluation.min().min(), evaluation.max().max())
    validation.add_shape(type="line", x0=low, y0=low, x1=high, y1=high,
                         line={"color": "gray", "dash": "dash"})
    st.plotly_chart(validation, width="stretch")
