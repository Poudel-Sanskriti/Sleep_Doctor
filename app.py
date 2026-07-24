"""Sleep Doctor — interactive demo of the project's two Linear Regression models.

Trains both models from sleep_health_dataset.csv on startup (cached, ~1s) so no
model files need to be committed. Mirrors notebooks/05_model_refinements.ipynb:
Path A (research-question) vs. the lifestyle/context extension.
"""

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

RANDOM_STATE = 117
DATA_PATH = "sleep_health_dataset.csv"
BUCKETS = ["Low", "Medium", "High"]

PRIMARY_FEATURES = ["age", "stress_score", "steps_that_day", "exercise_day"]

EXTENSION_NUMERIC = [
    "work_hours_that_day", "caffeine_mg_before_bed", "alcohol_units_before_bed",
    "screen_time_before_bed_mins", "bmi", "nap_duration_mins",
]
EXTENSION_BINARY = ["shift_work"]
EXTENSION_CATEGORICAL = [
    "chronotype", "mental_health_condition", "day_type", "season", "gender", "occupation",
]
CATEGORY_OPTIONS = {
    "chronotype": ["Neutral", "Evening", "Morning"],
    "mental_health_condition": ["Healthy", "Anxiety", "Depression", "Both"],
    "day_type": ["Weekday", "Weekend"],
    "season": ["Spring", "Summer", "Autumn", "Winter"],
    "gender": ["Female", "Male", "Other"],
    "occupation": [
        "Doctor", "Driver", "Freelancer", "Homemaker", "Lawyer", "Manager",
        "Nurse", "Retired", "Sales", "Software Engineer", "Student", "Teacher",
    ],
}

FRIENDLY_BASE = {
    "age": "Age",
    "stress_score": "Stress level",
    "steps_that_day": "Steps that day",
    "exercise_day": "Exercised today",
    "work_hours_that_day": "Work hours",
    "shift_work": "Shift work",
    "caffeine_mg_before_bed": "Caffeine before bed",
    "alcohol_units_before_bed": "Alcohol before bed",
    "screen_time_before_bed_mins": "Screen time before bed",
    "bmi": "BMI",
    "nap_duration_mins": "Nap duration",
    "chronotype": "Chronotype",
    "mental_health_condition": "Mental health",
    "day_type": "Day type",
    "season": "Season",
    "gender": "Gender",
    "occupation": "Occupation",
}

# Diverging pair from the project's chart palette: blue = raises the score, red = lowers it.
POSITIVE_COLOR = "#2a78d6"
NEGATIVE_COLOR = "#e34948"
STATUS_COLORS = {"Low": "#d03b3b", "Medium": "#fab219", "High": "#0ca30c"}
STATUS_ICONS = {"Low": "⚠️", "Medium": "●", "High": "✓"}


def to_bucket(scores: pd.Series) -> pd.Series:
    rounded = scores.round()
    return pd.cut(rounded, bins=[-np.inf, 4, 6, np.inf], labels=BUCKETS)


def dummy_label(dummy_col: str) -> str:
    for base in EXTENSION_CATEGORICAL:
        prefix = base + "_"
        if dummy_col.startswith(prefix):
            return f"{FRIENDLY_BASE[base]}: {dummy_col[len(prefix):]}"
    return dummy_col


@st.cache_resource
def train_models():
    df = pd.read_csv(DATA_PATH)
    df["quality_bucket"] = to_bucket(df["sleep_quality_score"])

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["quality_bucket"]
    )
    y_train = train_df["sleep_quality_score"]
    y_test = test_df["sleep_quality_score"]

    # Primary — the research question's three factors (activity = steps + exercise flag).
    X_primary_train = train_df[PRIMARY_FEATURES]
    X_primary_test = test_df[PRIMARY_FEATURES]
    primary_model = LinearRegression().fit(X_primary_train, y_train)
    primary_pred_test = primary_model.predict(X_primary_test)
    primary_r2 = r2_score(y_test, primary_pred_test)
    primary_acc = accuracy_score(
        test_df["quality_bucket"],
        to_bucket(pd.Series(primary_pred_test, index=X_primary_test.index)),
    )
    primary_means = X_primary_train.mean()

    # Extension — full lifestyle/context, one-hot encoded to match 05_model_refinements.
    extension_cols = PRIMARY_FEATURES + EXTENSION_NUMERIC + EXTENSION_BINARY + EXTENSION_CATEGORICAL
    X_all = pd.get_dummies(df[extension_cols], columns=EXTENSION_CATEGORICAL, drop_first=True)
    extension_columns = list(X_all.columns)

    X_ext_train = X_all.loc[train_df.index]
    X_ext_test = X_all.loc[test_df.index]
    extension_model = LinearRegression().fit(X_ext_train, y_train)
    extension_pred_test = extension_model.predict(X_ext_test)
    extension_r2 = r2_score(y_test, extension_pred_test)
    extension_acc = accuracy_score(
        test_df["quality_bucket"],
        to_bucket(pd.Series(extension_pred_test, index=X_ext_test.index)),
    )
    extension_means = X_ext_train.mean()

    return {
        "primary_model": primary_model,
        "primary_means": primary_means,
        "primary_r2": primary_r2,
        "primary_acc": primary_acc,
        "extension_model": extension_model,
        "extension_means": extension_means,
        "extension_columns": extension_columns,
        "extension_r2": extension_r2,
        "extension_acc": extension_acc,
    }


def predict_with_breakdown(model, means, x_row):
    """Predicted score + per-feature contribution vs. the training-set average person."""
    coefs = pd.Series(model.coef_, index=means.index)
    x_row = x_row.reindex(means.index).fillna(0.0)
    contributions = coefs * (x_row - means)
    pred = float(model.intercept_ + coefs.dot(x_row))
    return pred, contributions


ROW_HEIGHT = 34
BAR_THICKNESS = 20
LABEL_INK = "#52514e"


def contribution_chart(contributions: pd.Series, label_fn, top_n: int = 8) -> alt.Chart:
    ranked = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    ranked = ranked[ranked.abs() > 1e-6].head(top_n)
    plot_df = pd.DataFrame({
        "feature": [label_fn(f) for f in ranked.index],
        "contribution": ranked.values,
    })
    plot_df["direction"] = np.where(
        plot_df["contribution"] >= 0, "Raises predicted score", "Lowers predicted score"
    )
    order = plot_df["feature"].tolist()

    # Pad the x-domain so tip labels never get clipped at the chart edge.
    max_abs = max(plot_df["contribution"].abs().max(), 0.05) * 1.35
    y_enc = alt.Y(
        "feature:N",
        title=None,
        sort=order,
        scale=alt.Scale(paddingInner=0.4, paddingOuter=0.3),
        axis=alt.Axis(labelLimit=210, labelPadding=8, domain=False, ticks=False, labelFontSize=12),
    )

    zero_rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#c3c2b7").encode(x="x:Q")
    bars = (
        alt.Chart(plot_df)
        .mark_bar(size=BAR_THICKNESS, cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "contribution:Q",
                title="Effect on predicted score (points)",
                scale=alt.Scale(domain=[-max_abs, max_abs]),
            ),
            y=y_enc,
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Raises predicted score", "Lowers predicted score"],
                    range=[POSITIVE_COLOR, NEGATIVE_COLOR],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("contribution:Q", title="Effect (pts)", format="+.2f"),
            ],
        )
    )

    pos_df = plot_df[plot_df["contribution"] >= 0]
    neg_df = plot_df[plot_df["contribution"] < 0]
    pos_labels = (
        alt.Chart(pos_df)
        .mark_text(align="left", dx=6, fontSize=11, color=LABEL_INK)
        .encode(x="contribution:Q", y=y_enc, text=alt.Text("contribution:Q", format="+.2f"))
    )
    neg_labels = (
        alt.Chart(neg_df)
        .mark_text(align="right", dx=-6, fontSize=11, color=LABEL_INK)
        .encode(x="contribution:Q", y=y_enc, text=alt.Text("contribution:Q", format="+.2f"))
    )

    chart_height = ROW_HEIGHT * len(plot_df) + 20
    return (
        (zero_rule + bars + pos_labels + neg_labels)
        .properties(height=chart_height, padding={"left": 5, "right": 30, "top": 5, "bottom": 5})
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )


st.set_page_config(page_title="Sleep Doctor", page_icon="\U0001fa7a", layout="centered")
models = train_models()

st.title("\U0001fa7a\U0001f319 Sleep Doctor")
st.caption(
    "Based on age, stress, and activity level, can we predict sleep quality? "
    "An interactive demo of the project's Linear Regression models."
)

st.sidebar.header("Inputs")
mode = st.sidebar.radio(
    "Model",
    ["Research question (3 factors)", "Full lifestyle (extension)"],
    help="Primary mirrors the research question (age, stress, activity). "
         "Extension adds every lifestyle/context factor from the notebook's follow-up analysis.",
)
is_extension = mode == "Full lifestyle (extension)"

st.sidebar.subheader("Core factors")
age = st.sidebar.slider("Age", 18, 69, 35)
stress_score = st.sidebar.slider("Stress level (1 = calm, 10 = maxed out)", 1, 10, 5)
steps_that_day = st.sidebar.slider("Steps that day", 500, 20000, 7500, step=250)
exercise_day = st.sidebar.checkbox("Exercised today", value=False)

extension_inputs = {}
if is_extension:
    with st.sidebar.expander("Advanced: lifestyle & context", expanded=True):
        extension_inputs["work_hours_that_day"] = st.slider("Work hours that day", 0.0, 18.0, 7.0, 0.5)
        extension_inputs["shift_work"] = st.checkbox("Works a shift/night schedule")
        extension_inputs["caffeine_mg_before_bed"] = st.slider("Caffeine before bed (mg)", 0, 400, 40, 10)
        extension_inputs["alcohol_units_before_bed"] = st.slider("Alcohol before bed (units)", 0.0, 6.0, 0.5, 0.5)
        extension_inputs["screen_time_before_bed_mins"] = st.slider("Screen time before bed (min)", 0, 180, 60, 5)
        extension_inputs["bmi"] = st.slider("BMI", 16.0, 45.0, 26.0, 0.5)
        extension_inputs["nap_duration_mins"] = st.slider("Nap duration (min)", 0, 116, 15, 5)
        extension_inputs["chronotype"] = st.selectbox("Chronotype", CATEGORY_OPTIONS["chronotype"])
        extension_inputs["mental_health_condition"] = st.selectbox(
            "Mental health", CATEGORY_OPTIONS["mental_health_condition"]
        )
        extension_inputs["day_type"] = st.selectbox("Day type", CATEGORY_OPTIONS["day_type"])
        extension_inputs["season"] = st.selectbox("Season", CATEGORY_OPTIONS["season"])
        extension_inputs["gender"] = st.selectbox("Gender", CATEGORY_OPTIONS["gender"])
        extension_inputs["occupation"] = st.selectbox("Occupation", CATEGORY_OPTIONS["occupation"])

raw_inputs = {
    "age": age,
    "stress_score": stress_score,
    "steps_that_day": steps_that_day,
    "exercise_day": int(exercise_day),
}

if is_extension:
    raw_inputs.update(extension_inputs)
    raw_inputs["shift_work"] = int(extension_inputs["shift_work"])
    input_row = pd.DataFrame([raw_inputs])
    dummies = pd.get_dummies(input_row, columns=EXTENSION_CATEGORICAL)
    x_row = dummies.iloc[0].reindex(models["extension_columns"]).fillna(0.0)
    pred_score, contributions = predict_with_breakdown(
        models["extension_model"], models["extension_means"], x_row
    )
    r2, acc = models["extension_r2"], models["extension_acc"]
    label_fn = dummy_label
else:
    x_row = pd.Series(raw_inputs)[PRIMARY_FEATURES]
    pred_score, contributions = predict_with_breakdown(
        models["primary_model"], models["primary_means"], x_row
    )
    r2, acc = models["primary_r2"], models["primary_acc"]
    label_fn = lambda f: FRIENDLY_BASE.get(f, f)

bucket = to_bucket(pd.Series([pred_score])).iloc[0]
pred_display = float(np.clip(pred_score, 1, 10))

col1, col2 = st.columns(2)
col1.metric("Predicted sleep quality", f"{pred_display:.1f} / 10")
col2.markdown(
    f"**Predicted class**<br>"
    f"<span style='font-size:1.6rem;color:{STATUS_COLORS[bucket]}'>{STATUS_ICONS[bucket]} {bucket}</span>",
    unsafe_allow_html=True,
)

st.subheader("What drove this prediction")
st.caption(
    "Each bar is that input's pull on the score, versus an average person in the dataset "
    "(Linear Regression coefficient x how far your input sits from the dataset mean)."
)
st.markdown(
    f"<span style='color:{POSITIVE_COLOR}'>⬤</span> Raises predicted score"
    f"&nbsp;&nbsp;&nbsp;"
    f"<span style='color:{NEGATIVE_COLOR}'>⬤</span> Lowers predicted score",
    unsafe_allow_html=True,
)
st.altair_chart(contribution_chart(contributions, label_fn), width="stretch")

st.divider()
st.caption(
    f"**Honesty check:** this {'extension' if is_extension else 'research-question'} model "
    f"explains about **{r2:.0%}** of the variation in sleep quality (R²) and sorts people into "
    f"Low/Medium/High correctly about **{acc:.0%}** of the time on held-out data. "
    "Trained on a synthetic Kaggle dataset (100,000 rows) — these are patterns built into the "
    "data generator, not verified facts about human sleep."
)
