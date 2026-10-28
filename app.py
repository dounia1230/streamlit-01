"""Exploration Auto MPG et comparaison de plusieurs modèles de régression."""

import pandas as pd
import plotly.express as px
import streamlit as st

from modeling import (
    BASELINE, CV_MAE, CV_STD, FEATURES, NUMERIC_FEATURES, PREDICTION, TARGET,
    TEST_MAE, TEST_R2, TEST_RMSE, compare_predictions, train_models,
)

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
COLORS = {"usa": "#2563EB", "japan": "#93C5FD", "europe": "#EF4444"}
LABELS = {
    "name": "Véhicule", "origin": "Origine", "model_year": "Année du modèle",
    "cylinders": "Cylindres", "displacement": "Cylindrée (in³)",
    "horsepower": "Puissance (hp)", "weight_kg": "Poids (kg)",
    "acceleration": "Accélération (s)", "mpg": "Efficacité (mpg)",
    TARGET: "Consommation (L/100 km)",
}
DISPLAY_COLUMNS = ["name", "origin", "model_year", "mpg", TARGET] + NUMERIC_FEATURES[:-1]


@st.cache_data(show_spinner=False)
def load_data():
    data = pd.read_csv(DATA_URL)
    data["weight_kg"] = data["weight"] * 0.45359237
    data[TARGET] = 235.214583 / data["mpg"]
    data["model_year"] = data["model_year"].map(lambda year: year + 1900 if year < 100 else year)
    return data


@st.cache_resource(show_spinner="Entraînement et validation croisée des modèles…")
def cached_training(data):
    return train_models(data)


def show_chart(chart):
    chart.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(chart, width="stretch")


def download_csv(label, frame, filename, key):
    st.download_button(
        label, frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename, mime="text/csv", key=key,
    )


def exploration_filters(data):
    st.sidebar.header("Filtres d’exploration")
    origin = st.sidebar.selectbox(
        "Origine", ["Toutes"] + sorted(data["origin"].dropna().unique()), key="filter_origin",
    )
    cylinders = sorted(data["cylinders"].dropna().unique().tolist())
    selected = st.sidebar.multiselect(
        "Nombre de cylindres", cylinders, default=cylinders, key="filter_cylinders",
    )
    years = st.sidebar.slider(
        "Année du modèle", int(data["model_year"].min()), int(data["model_year"].max()),
        (int(data["model_year"].min()), int(data["model_year"].max())), key="filter_years",
    )
    min_mpg = st.sidebar.slider(
        "Efficacité minimale (mpg)", float(data["mpg"].min()), float(data["mpg"].max()),
        float(data["mpg"].min()), step=0.5, key="filter_mpg",
    )
    filtered = data[
        data["cylinders"].isin(selected)
        & data["model_year"].between(*years)
        & (data["mpg"] >= min_mpg)
    ].copy()
    if origin != "Toutes":
        filtered = filtered[filtered["origin"] == origin]
    st.sidebar.caption(
        "Ces filtres modifient les graphiques et les statistiques. "
        "Les modèles apprennent toujours sur le même jeu de données complet."
    )
    return filtered


def render_overview(filtered, total):
    st.subheader("Indicateurs et véhicules")
    st.caption(f"{len(filtered)} véhicules sélectionnés sur {total}")
    if filtered.empty:
        st.warning("Aucun véhicule ne correspond aux filtres. Élargis la sélection à gauche.")
        return
    columns = st.columns(4)
    columns[0].metric("Efficacité moyenne", f"{filtered['mpg'].mean():.1f} mpg")
    columns[1].metric("Consommation moyenne", f"{filtered[TARGET].mean():.2f} L/100 km")
    columns[2].metric("Consommation médiane", f"{filtered[TARGET].median():.2f} L/100 km")
    columns[3].metric("Poids moyen", f"{filtered['weight_kg'].mean():.0f} kg")
    st.caption("La consommation moyenne est la moyenne des conversions individuelles des MPG.")
    st.dataframe(filtered[DISPLAY_COLUMNS].rename(columns=LABELS).round(2), width="stretch", hide_index=True)
    download_csv("Télécharger la sélection", filtered[DISPLAY_COLUMNS], "vehicules.csv", "download_data")
    selected_index = st.selectbox(
        "Examiner un véhicule", filtered.index.tolist(), key="selected_vehicle",
        format_func=lambda i: f"{filtered.loc[i, 'name']} — {int(filtered.loc[i, 'model_year'])} (ligne {i})",
    )
    st.dataframe(filtered.loc[[selected_index], DISPLAY_COLUMNS].rename(columns=LABELS).round(2),
                 width="stretch", hide_index=True)
    show_chart(px.scatter(
        filtered, x="weight_kg", y=TARGET, color="origin", hover_name="name",
        color_discrete_map=COLORS, labels=LABELS, title="Poids et consommation",
    ))
    left, right = st.columns(2)
    with left:
        yearly = filtered.groupby("model_year", as_index=False)[TARGET].mean()
        show_chart(px.line(yearly, x="model_year", y=TARGET, markers=True,
                           labels=LABELS, title="Consommation moyenne par année"))
    with right:
        counts = filtered["origin"].value_counts().rename_axis("origin").reset_index(name="Véhicules")
        show_chart(px.bar(counts, x="origin", y="Véhicules", color="origin",
                          color_discrete_map=COLORS, labels=LABELS, title="Répartition par origine"))


def render_statistics(filtered):
    st.subheader("Statistiques descriptives")
    if filtered.empty:
        st.info("Sélectionne au moins un véhicule pour afficher les statistiques.")
        return
    stats_columns = ["mpg", TARGET] + NUMERIC_FEATURES
    stats = filtered[stats_columns].describe().T.rename(
        columns={"count": "Valeurs présentes", "mean": "Moyenne", "std": "Écart-type",
                 "min": "Minimum", "25%": "Q1 (25 %)", "50%": "Médiane", "75%": "Q3 (75 %)", "max": "Maximum"},
        index=LABELS,
    )
    stats["Valeurs absentes"] = filtered[stats_columns].isna().sum().rename(index=LABELS)
    st.dataframe(stats.round(2), width="stretch")
    st.caption("L’écart-type mesure la dispersion. Q1 et Q3 encadrent la moitié centrale des valeurs.")
    download_csv("Télécharger les statistiques", stats.rename_axis("Variable").reset_index(),
                 "statistiques.csv", "download_stats")
    variable = st.selectbox("Variable à explorer", stats_columns, format_func=LABELS.get, key="stats_variable")
    left, right = st.columns(2)
    with left:
        show_chart(px.histogram(
            filtered, x=variable, color="origin", nbins=20, barmode="overlay", opacity=0.7,
            color_discrete_map=COLORS, labels=LABELS, title="Distribution des valeurs",
        ))
    with right:
        show_chart(px.box(filtered, x="origin", y=variable, color="origin", points="outliers",
                          color_discrete_map=COLORS, labels=LABELS, title="Dispersion par origine"))
    st.subheader("Comparaison par origine")
    grouped = filtered.groupby("origin").agg(
        Véhicules=(TARGET, "size"),
        Consommation_moyenne=(TARGET, "mean"),
        Consommation_médiane=(TARGET, "median"),
        Écart_type=(TARGET, "std"),
        MPG_moyen=("mpg", "mean"),
        Poids_moyen_kg=("weight_kg", "mean"),
    ).rename(columns={
        "Consommation_moyenne": "Moyenne (L/100 km)", "Consommation_médiane": "Médiane (L/100 km)",
        "Écart_type": "Écart-type (L/100 km)", "MPG_moyen": "Moyenne (mpg)", "Poids_moyen_kg": "Poids moyen (kg)",
    })
    st.dataframe(grouped.round(2), width="stretch")
    st.subheader("Corrélations entre les variables numériques")
    if len(filtered) < 3:
        st.info("Sélectionne au moins trois véhicules pour afficher les corrélations.")
        return
    method = st.radio("Méthode de corrélation", ["Pearson", "Spearman"], horizontal=True)
    correlation = filtered[stats_columns].rename(columns=LABELS).corr(method=method.lower())
    # Les colonnes constantes ne définissent pas de coefficient de corrélation.
    correlation = correlation.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if correlation.empty:
        st.info("Les variables sélectionnées sont constantes : corrélations non définies.")
    else:
        show_chart(px.imshow(correlation, text_auto=".2f", zmin=-1, zmax=1,
                             color_continuous_scale="RdBu_r", aspect="auto", title=f"Corrélations de {method}"))
    st.caption(
        "Une corrélation n’établit pas une causalité. La consommation est calculée à partir "
        "des MPG : leur relation est donc mécanique. Les MPG ne sont jamais utilisés pour entraîner les modèles. "
        "Les colonnes constantes sont omises ; les cellules sans coefficient défini restent vides."
    )


def render_models(comparison):
    st.subheader("Comparer les performances")
    st.write(
        f"**{comparison.train_count} véhicules pour l’entraînement**, "
        f"**{comparison.test_count} pour le test**, identiques pour tous les modèles. "
        "La validation croisée utilise cinq partitions du jeu d’entraînement."
    )
    st.success(f"Modèle recommandé par validation croisée : **{comparison.recommended}**")
    baseline_score = comparison.scores.loc[comparison.scores["Modèle"] == BASELINE, CV_MAE].iloc[0]
    best_score = comparison.scores.loc[comparison.scores["Modèle"] == comparison.recommended, CV_MAE].iloc[0]
    if best_score >= baseline_score:
        st.warning("Le modèle recommandé ne dépasse pas la référence médiane en validation croisée.")
    st.dataframe(comparison.scores.round(3), hide_index=True, width="stretch")
    st.caption(
        "Classement selon la MAE de validation croisée : plus basse = mieux. "
        "La référence médiane prédit toujours la médiane du jeu d’entraînement. "
        "MAE = erreur absolue moyenne ; RMSE pénalise davantage les grandes erreurs. "
        "Pour R², 1 est idéal et un score négatif est possible. "
        "L’écart-type décrit la variation de la MAE entre les cinq partitions."
    )
    download_csv("Télécharger les scores", comparison.scores, "comparaison_modeles.csv", "download_scores")
    show_chart(px.bar(
        comparison.scores, x=CV_MAE, y="Modèle", orientation="h", error_x=CV_STD,
        title="Erreur de validation croisée et dispersion entre partitions",
    ).update_yaxes(autorange="reversed"))
    with st.expander("Méthode et familles de modèles"):
        st.write(
            "Ridge : régression linéaire régularisée. KNN : véhicules voisins. Random Forest "
            "et Extra Trees : ensembles d’arbres. Gradient Boosting : arbres qui corrigent "
            "progressivement les erreurs. Les réglages sont fixes pour cette comparaison."
        )
        st.write(
            "La médiane de remplacement, la normalisation des nombres et l’encodage de l’origine "
            "sont appris dans chaque partition d’entraînement. Le choix recommandé exclut la référence "
            "et repose uniquement sur la validation croisée. Le jeu de test reste réservé à l’évaluation. "
            "Les modèles utilisés pour prédire sont ceux évalués ici, entraînés sur 80 % des véhicules. "
            "Ces résultats concernent des voitures historiques et une séparation aléatoire, "
            "pas la capacité à prédire une nouvelle génération de véhicules."
        )
    st.subheader("Examiner les erreurs sur le jeu de test")
    model_name = st.selectbox(
        "Modèle à examiner", comparison.scores["Modèle"].tolist(), key="diagnostic_model",
        index=comparison.scores["Modèle"].tolist().index(comparison.recommended),
    )
    score = comparison.scores.set_index("Modèle").loc[model_name]
    a, b, c = st.columns(3)
    a.metric("MAE test", f"{score[TEST_MAE]:.2f} L/100 km")
    b.metric("RMSE test", f"{score[TEST_RMSE]:.2f} L/100 km")
    c.metric("R² test", f"{score[TEST_R2]:.3f}")
    evaluation = comparison.evaluation[["Réel (L/100 km)", model_name]].rename(columns={model_name: PREDICTION})
    evaluation["Résidu (réel − prédit)"] = evaluation["Réel (L/100 km)"] - evaluation[PREDICTION]
    left, right = st.columns(2)
    with left:
        chart = px.scatter(evaluation, x="Réel (L/100 km)", y=PREDICTION, title="Réel et prédit")
        low = float(evaluation.iloc[:, :2].min().min())
        high = float(evaluation.iloc[:, :2].max().max())
        chart.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line={"color": "gray", "dash": "dash"})
        show_chart(chart)
    with right:
        chart = px.scatter(evaluation, x=PREDICTION, y="Résidu (réel − prédit)", title="Résidus : au plus près de zéro")
        chart.add_hline(y=0, line_dash="dash", line_color="gray")
        show_chart(chart)
    with st.expander("Voir toutes les prédictions de test"):
        st.dataframe(comparison.evaluation.round(2), width="stretch")
        download_csv("Télécharger les prédictions de test", comparison.evaluation,
                     "predictions_test.csv", "download_test")


def render_prediction(data, comparison):
    st.subheader("Une voiture, tous les modèles")
    st.write("Saisis les caractéristiques puis clique sur Prédire pour voir tous les modèles et le meilleur résultat de validation.")
    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            cylinders = st.selectbox("Nombre de cylindres", sorted(data["cylinders"].unique().tolist()), index=1, key="car_cylinders")
            displacement = st.number_input("Cylindrée (in³)", min_value=50.0, max_value=500.0, value=150.0, step=5.0)
            horsepower = st.number_input("Puissance (hp)", min_value=30.0, max_value=250.0, value=100.0, step=5.0)
            origin = st.selectbox("Origine", ["usa", "europe", "japan"], key="car_origin")
        with right:
            weight_kg = st.number_input("Poids (kg)", min_value=500.0, max_value=2500.0, value=1200.0, step=25.0, key="car_weight")
            acceleration = st.number_input("Accélération (secondes)", min_value=8.0, max_value=25.0, value=15.0, step=0.5)
            model_year = st.slider("Année du modèle", int(data["model_year"].min()), int(data["model_year"].max()), 1976, key="car_year")
        submitted = st.form_submit_button("Prédire", type="primary")
    if submitted:
        car = pd.DataFrame([{
            "cylinders": cylinders, "displacement": displacement, "horsepower": horsepower,
            "weight_kg": weight_kg, "acceleration": acceleration,
            "model_year": model_year, "origin": origin,
        }])
        st.session_state["prediction_car"] = car
    if "prediction_car" not in st.session_state:
        st.info("Clique sur Prédire : les cinq modèles, leurs estimations et le meilleur modèle s’afficheront ici.")
        return
    car = st.session_state["prediction_car"]
    results = compare_predictions(comparison, car)
    st.markdown("#### Dernière simulation validée")
    st.dataframe(car[FEATURES].rename(columns=LABELS), hide_index=True, width="stretch")
    outside = [LABELS[col] for col in NUMERIC_FEATURES
               if not data[col].min() <= car[col].iloc[0] <= data[col].max()]
    if outside:
        st.warning("Valeurs en dehors des plages observées : " + ", ".join(outside) + ". Ces estimations extrapolent les données.")
    chosen = results.loc[results["Modèle"] == comparison.recommended].iloc[0]
    estimates = results.loc[results["Modèle"] != BASELINE, PREDICTION]
    st.success(f"🏆 Meilleur modèle en validation croisée : **{comparison.recommended}**")
    st.write(
        f"Ce modèle obtient la plus faible erreur moyenne parmi les cinq modèles : "
        f"**{chosen[CV_MAE]:.2f} L/100 km** en validation croisée. "
        "Cette comparaison mesure la précision sur les véhicules de validation."
    )
    a, b, c = st.columns(3)
    a.metric("Estimation recommandée", f"{chosen[PREDICTION]:.2f} L/100 km")
    b.metric("Équivalent", f"{chosen['Équivalent (mpg)']:.1f} mpg" if chosen[PREDICTION] > 0 else "Non défini")
    c.metric("Écart entre les cinq modèles", f"{estimates.max() - estimates.min():.2f} L/100 km")
    st.caption(
        f"Modèle recommandé : {comparison.recommended}. L’écart est le maximum moins le minimum "
        "des cinq estimations, sans la référence ; ce n’est pas un intervalle de confiance. "
        "Les erreurs de test décrivent une performance globale, pas l’erreur connue pour cette voiture."
    )
    if (results[PREDICTION] <= 0).any():
        st.warning("Une estimation est nulle ou négative et n’a pas de sens physique ; son équivalent MPG est omis.")
    st.markdown("#### Prédictions de tous les modèles")
    shown = results[["Modèle", PREDICTION, "Équivalent (mpg)", "Recommandé (validation croisée)", CV_MAE, TEST_MAE, TEST_RMSE, TEST_R2]]
    st.dataframe(shown.round(3), hide_index=True, width="stretch")
    show_chart(px.bar(
        shown, x="Modèle", y=PREDICTION, color="Recommandé (validation croisée)",
        color_discrete_map={True: "#2563EB", False: "#94A3B8"},
        text_auto=".2f", title="Consommation estimée pour la même voiture",
    ))
    download_csv("Télécharger cette comparaison", car.merge(shown, how="cross"),
                 "comparaison_predictions.csv", "download_predictions")
    st.caption("Estimations sur des voitures de 1970 à 1982 ; elles ne constituent pas des mesures réelles.")


def main():
    st.set_page_config(page_title="Auto MPG — Statistiques et modèles", page_icon="🚗", layout="wide")
    st.title("🚗 Exploration des données automobiles")
    st.caption("Auto MPG · 1970–1982 · Statistiques, apprentissage et comparaison des prédictions")
    st.info("MPG élevé = plus économe. L/100 km faible = plus économe.")
    try:
        data = load_data()
    except Exception as error:
        st.error("Impossible de charger le CSV. Vérifie ta connexion Internet et réessaie.")
        st.exception(error)
        st.stop()
    filtered = exploration_filters(data)
    prediction, models, statistics, overview = st.tabs([
        "Prédire", "Comparer les modèles", "Statistiques", "Vue d’ensemble",
    ])
    comparison = cached_training(data)
    with prediction:
        render_prediction(data, comparison)
    with models:
        render_models(comparison)
    with overview:
        render_overview(filtered, len(data))
    with statistics:
        render_statistics(filtered)


if __name__ == "__main__":
    main()
