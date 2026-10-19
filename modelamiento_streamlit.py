"""
EC1 - Modelamiento Predictivo de Datos - Grupo 8
Caso adaptado: Variabilidad e incumplimiento en la entrega de pedidos
de e-commerce y su impacto en la satisfacción del cliente y los costos logísticos.
Dataset: Brazilian E-Commerce Public Dataset by Olist (consolidado a nivel de pedido).

Ejecutar con:  streamlit run app.py
Requiere 'olist_consolidado_pedido.csv' o 'olist_consolidado_pedido.xlsx' en la misma carpeta.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
sns.set_style("whitegrid")

# ============================================================
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Grupo 8 - Modelamiento Predictivo | Olist",
    layout="wide",
)

st.title("📦 Variabilidad e incumplimiento en la entrega de pedidos de e-commerce")
st.caption(
    "Caso adaptado — Curso Modelamiento Predictivo de Datos | "
    "Dataset: Brazilian E-Commerce Public Dataset by Olist"
)

# ============================================================
# CARGA DE DATOS (cacheada para no recargar en cada interacción)
# ============================================================
@st.cache_data
def cargar_datos():
    columnas_fecha = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    if os.path.exists("olist_consolidado_pedido.csv"):
        df = pd.read_csv("olist_consolidado_pedido.csv", parse_dates=columnas_fecha)
    elif os.path.exists("olist_consolidado_pedido.xlsx"):
        df = pd.read_excel("olist_consolidado_pedido.xlsx", parse_dates=columnas_fecha)
    elif os.path.exists("olist_consolidado_pedido(1).xlsx"):
        df = pd.read_excel("olist_consolidado_pedido(1).xlsx", parse_dates=columnas_fecha)
    else:
        raise FileNotFoundError
    return df

try:
    df = cargar_datos()
except FileNotFoundError:
    st.error(
        "⚠️ No se encontró el archivo de datos. Sube a GitHub o coloca junto al script "
        "'olist_consolidado_pedido.csv', 'olist_consolidado_pedido.xlsx' o "
        "'olist_consolidado_pedido(1).xlsx'."
    )
    st.stop()

# Subconjunto válido para OTD / variabilidad: pedidos efectivamente entregados
df_entregados = df[df["late_delivery"].notna()].copy()
df_entregados["late_delivery"] = df_entregados["late_delivery"].astype(int)

# Subconjunto válido para satisfacción (Dimensión 4): entregados + con reseña
df_kpi4 = df_entregados[df_entregados["review_score"].notna()].copy()
df_kpi4["satisfaccion_bin"] = (df_kpi4["review_score"] >= 4).astype(int)

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ OTD — Redes Neuronales",
    "2️⃣ Variabilidad — Series de Tiempo",
    "3️⃣ Sobrecosto — Clustering",
    "4️⃣ Satisfacción — Árbol / Random Forest",
])

# ============================================================
# TAB 1 - DIMENSIÓN 1: CUMPLIMIENTO DE ENTREGAS (OTD)
# REDES NEURONALES - PERCEPTRÓN MULTICAPA (MLP)
# ============================================================

with tab1:

    st.header("Dimensión 1: Cumplimiento de entregas (OTD)")

    st.markdown(
        "**Indicador:** OTD (On-Time Delivery)  \n"
        "**Técnica:** Redes neuronales - Perceptrón Multicapa (MLP)  \n"
        "**Proceso asociado:** Planificación y control de la entrega"
    )

    # --------------------------------------------------------
    # 1. PREPARACIÓN DE LOS DATOS
    # --------------------------------------------------------

    df_modelo = df[df["late_delivery"].notna()].copy()

    # Crear variable objetivo
    df_modelo["OTD"] = 1 - df_modelo["late_delivery"]

    variables = [
        "freight_value",
        "price",
        "payment_value_total",
        "payment_installments_max",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "product_photos_qty",
        "n_items",
        "n_sellers",
        "n_products",
        "payment_type_main",
        "seller_state",
        "customer_state"
    ]

    X = df_modelo[variables].copy()
    y = df_modelo["OTD"].copy()

    # --------------------------------------------------------
    # 2. DISTRIBUCIÓN DEL OTD
    # --------------------------------------------------------

    st.subheader("Distribución del cumplimiento de entregas")

    otd_counts = y.value_counts().sort_index()

    st.bar_chart(
        pd.DataFrame({
            "Pedidos": otd_counts.values
        }, index=["Fuera de plazo", "Dentro de plazo"])
    )

    # --------------------------------------------------------
    # 3. DIVISIÓN DE DATOS
    # --------------------------------------------------------

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    num = X.select_dtypes(include="number").columns
    cat = X.select_dtypes(exclude="number").columns

    preprocesador = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            num
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            cat
        )
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    X_train = preprocesador.fit_transform(X_train)
    X_test = preprocesador.transform(X_test)

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    # --------------------------------------------------------
    # 4. MODELO DE RED NEURONAL
    # --------------------------------------------------------

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    model = Sequential([
        Dense(
            64,
            activation="relu",
            input_shape=(X_train.shape[1],)
        ),
        Dropout(0.30),

        Dense(
            32,
            activation="relu"
        ),
        Dropout(0.30),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="AUC")
        ]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # --------------------------------------------------------
    # 5. ENTRENAMIENTO
    # --------------------------------------------------------

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=50,
        batch_size=256,
        callbacks=[early_stop],
        verbose=0
    )

    # --------------------------------------------------------
    # 6. PREDICCIÓN
    # --------------------------------------------------------

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix
    )

    y_prob = model.predict(
        X_test,
        verbose=0
    ).ravel()

    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # --------------------------------------------------------
    # 7. RESULTADOS
    # --------------------------------------------------------

    st.subheader("Resultados del modelo predictivo")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("Precision", f"{precision:.2%}")
    col3.metric("Recall", f"{recall:.2%}")
    col4.metric("F1-score", f"{f1:.2%}")
    col5.metric("AUC-ROC", f"{auc:.2%}")

    # --------------------------------------------------------
    # 8. MATRIZ DE CONFUSIÓN
    # --------------------------------------------------------

    st.subheader("Matriz de confusión")

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Fuera de plazo", "Dentro de plazo"],
        columns=["Predicho fuera", "Predicho dentro"]
    )

    st.dataframe(cm_df)

    # --------------------------------------------------------
    # 9. IMPORTANCIA DE VARIABLES
    # --------------------------------------------------------

    st.subheader("Importancia de variables")

    import numpy as np

    base_auc = roc_auc_score(
        y_test,
        model.predict(X_test, verbose=0).ravel()
    )

    importancias = []

    for i in range(X_test.shape[1]):

        X_perm = X_test.copy()

        np.random.seed(42)
        X_perm[:, i] = np.random.permutation(
            X_perm[:, i]
        )

        auc_perm = roc_auc_score(
            y_test,
            model.predict(
                X_perm,
                verbose=0
            ).ravel()
        )

        importancias.append(
            base_auc - auc_perm
        )

    nombres = preprocesador.get_feature_names_out()

    importancia_df = pd.DataFrame({
        "Variable": nombres,
        "Importancia": importancias
    })

    importancia_df = (
        importancia_df
        .sort_values(
            "Importancia",
            ascending=False
        )
        .head(10)
    )

    st.bar_chart(
        importancia_df.set_index("Variable")
    )

    # --------------------------------------------------------
    # 10. PREDICCIÓN INTERACTIVA
    # --------------------------------------------------------

    st.subheader("Predicción de cumplimiento de entrega")

    st.write(
        "Ingrese las características del pedido para estimar "
        "la probabilidad de cumplimiento del plazo."
    )

    st.info(
        "El modelo genera una probabilidad estimada de que "
        "el pedido sea entregado dentro del plazo establecido."
    )

    st.markdown(
        "**Modelo:** Perceptrón Multicapa (MLP)  \n"
        "**Arquitectura:** 64 → 32 → 1 neuronas  \n"
        "**Activaciones:** ReLU → ReLU → Sigmoid  \n"
        "**Dropout:** 0.30  \n"
        "**Optimizador:** Adam"
    )

# ============================================================
# TAB 2 — DIMENSIÓN 2: Variabilidad mensual de los días reales
# Técnica: series de tiempo — ARIMA(1,0,0)
# ============================================================

with tab2:
    st.header("Dimensión 2: Variabilidad del tiempo real de entrega")
    st.markdown(
        "Analiza cómo cambia mes a mes la dispersión de los días que tardan "
        "los pedidos en llegar al cliente y estima su comportamiento futuro."
    )
    st.info(
        "**Indicador utilizado:** desviación estándar mensual de los días reales de entrega. "
        "No es la diferencia entre la fecha real y la fecha estimada."
    )

    # --------------------------------------------------------
    # 1. Preparar la serie correcta
    # --------------------------------------------------------
    required_cols = [
        "order_status", "order_purchase_timestamp",
        "order_delivered_customer_date"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Faltan columnas necesarias para la Dimensión 2: {', '.join(missing_cols)}")
        st.stop()

    df2 = df.copy()
    df2["order_purchase_timestamp"] = pd.to_datetime(
        df2["order_purchase_timestamp"], errors="coerce"
    )
    df2["order_delivered_customer_date"] = pd.to_datetime(
        df2["order_delivered_customer_date"], errors="coerce"
    )

    df2 = df2[
        (df2["order_status"].astype(str).str.lower() == "delivered")
        & df2["order_purchase_timestamp"].notna()
        & df2["order_delivered_customer_date"].notna()
    ].copy()

    df2["dias_entrega_real"] = (
        df2["order_delivered_customer_date"] - df2["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # Excluir duraciones no válidas; las entregas reales no pueden tener duración negativa.
    df2 = df2[df2["dias_entrega_real"].notna() & (df2["dias_entrega_real"] >= 0)].copy()
    df2["mes"] = df2["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()

    mensual = (
        df2.groupby("mes")["dias_entrega_real"]
        .agg(variabilidad_dias="std", promedio_dias="mean", pedidos="count")
        .reset_index()
        .sort_values("mes")
    )
    mensual = mensual.dropna(subset=["variabilidad_dias"])

    # El trabajo utiliza la serie entre enero de 2017 y agosto de 2018 (20 meses).
    serie_periodo = mensual[
        (mensual["mes"] >= pd.Timestamp("2017-01-01"))
        & (mensual["mes"] <= pd.Timestamp("2018-08-01"))
    ].copy()
    serie = serie_periodo.set_index("mes")["variabilidad_dias"].astype(float).dropna().sort_index()

    if len(serie) < 13:
        st.error(
            "No hay suficientes meses en el archivo para evaluar ARIMA con la división "
            "de entrenamiento y prueba utilizada en el informe. Revisa el dataset."
        )
        st.stop()

    # --------------------------------------------------------
    # 2. Filtros y resumen descriptivo
    # --------------------------------------------------------
    st.subheader("Panorama de la variabilidad mensual")
    col_filtro, col_info = st.columns([1, 2])
    with col_filtro:
        vista = st.selectbox(
            "Periodo mostrado", 
            ["Todo el periodo (ene. 2017–ago. 2018)", "Últimos 12 meses", "Año 2017", "Año 2018"],
            key="d2_periodo_vista"
        )
    if vista == "Últimos 12 meses":
        serie_vista = serie.tail(12)
    elif vista == "Año 2017":
        serie_vista = serie[serie.index.year == 2017]
    elif vista == "Año 2018":
        serie_vista = serie[serie.index.year == 2018]
    else:
        serie_vista = serie

    with col_info:
        st.caption(
            f"Se calcularon **{len(serie)} observaciones mensuales** a partir de "
            f"{len(df2):,} pedidos entregados válidos. La serie utilizada para el modelo "
            "abarca enero de 2017 a agosto de 2018."
        )

    media_var = serie.mean()
    desv_var = serie.std(ddof=1)
    mes_max = serie.idxmax().strftime("%B %Y")
    mes_min = serie.idxmin().strftime("%B %Y")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Variabilidad mensual promedio", f"{media_var:.2f} días")
    k2.metric("Desviación estándar de la serie", f"{desv_var:.2f} días")
    k3.metric("Mayor variabilidad", f"{serie.max():.2f} días", mes_max)
    k4.metric("Menor variabilidad", f"{serie.min():.2f} días", mes_min)

    try:
        import plotly.express as px
        import plotly.graph_objects as go
        plotly_disponible = True
    except ImportError:
        plotly_disponible = False

    if plotly_disponible:
        fig_hist = px.line(
            x=serie_vista.index, y=serie_vista.values, markers=True,
            labels={"x": "Mes de compra", "y": "Desviación estándar (días)"},
            title="Evolución mensual de la variabilidad de los días reales de entrega"
        )
        fig_hist.update_traces(line=dict(width=3), marker=dict(size=8), hovertemplate="%{x|%b %Y}<br>Variabilidad: %{y:.2f} días<extra></extra>")
        fig_hist.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        fig_hist, ax_hist = plt.subplots(figsize=(11, 4))
        ax_hist.plot(serie_vista.index, serie_vista.values, marker="o", linewidth=2)
        ax_hist.set_title("Evolución mensual de la variabilidad de los días reales de entrega")
        ax_hist.set_xlabel("Mes de compra")
        ax_hist.set_ylabel("Desviación estándar (días)")
        ax_hist.grid(True, alpha=0.3)
        fig_hist.tight_layout()
        st.pyplot(fig_hist)
        plt.close(fig_hist)

    with st.expander("Ver tabla de la serie mensual"):
        tabla_serie = serie_periodo.copy()
        tabla_serie["mes"] = tabla_serie["mes"].dt.strftime("%Y-%m")
        tabla_serie = tabla_serie.rename(columns={
            "mes": "Mes", "variabilidad_dias": "Variabilidad (días)",
            "promedio_dias": "Promedio de entrega (días)", "pedidos": "Pedidos entregados"
        })
        st.dataframe(
            tabla_serie.style.format({
                "Variabilidad (días)": "{:.3f}",
                "Promedio de entrega (días)": "{:.3f}",
                "Pedidos entregados": "{:,.0f}"
            }), use_container_width=True, hide_index=True
        )
        st.download_button(
            "Descargar serie mensual (CSV)",
            data=tabla_serie.to_csv(index=False).encode("utf-8-sig"),
            file_name="serie_variabilidad_mensual_olist.csv",
            mime="text/csv",
            key="d2_download_serie"
        )

    # --------------------------------------------------------
    # 3. Comparación de modelos: mismos periodos del trabajo
    # --------------------------------------------------------
    st.divider()
    st.subheader("Evaluación y selección del modelo")
    st.markdown(
        "Para mantener la evaluación alineada con el análisis realizado, se utilizan "
        "las primeras 12 observaciones (enero–diciembre de 2017) para entrenamiento "
        "y las últimas 8 (enero–agosto de 2018) para prueba."
    )

    train = serie.iloc[:12]
    test = serie.iloc[12:]
    ordenes = [(0, 0, 1), (1, 0, 0), (1, 1, 1), (1, 1, 0), (0, 1, 1)]
    resultados_modelos = []
    modelos_ajustados = {}

    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    for orden in ordenes:
        nombre = f"ARIMA{orden}"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ajustado = ARIMA(train, order=orden).fit()
                pred = np.asarray(ajustado.forecast(steps=len(test)), dtype=float)
            modelos_ajustados[nombre] = ajustado
            resultados_modelos.append({
                "Modelo": nombre,
                "MAE": mean_absolute_error(test.values, pred),
                "MSE": mean_squared_error(test.values, pred),
                "RMSE": np.sqrt(mean_squared_error(test.values, pred))
            })
        except Exception as exc:
            st.warning(f"No se pudo ajustar {nombre}: {exc}")

    # Modelos de referencia
    pred_naive = np.repeat(float(train.iloc[-1]), len(test))
    resultados_modelos.append({
        "Modelo": "Naive", "MAE": mean_absolute_error(test.values, pred_naive),
        "MSE": mean_squared_error(test.values, pred_naive),
        "RMSE": np.sqrt(mean_squared_error(test.values, pred_naive))
    })

    tabla_modelos = pd.DataFrame(resultados_modelos).sort_values("RMSE").reset_index(drop=True)
    if tabla_modelos.empty or "ARIMA(1, 0, 0)" not in modelos_ajustados:
        st.error("No se pudo ajustar el modelo ARIMA(1,0,0). Revisa la serie y las dependencias.")
        st.stop()

    modelo_seleccionado = modelos_ajustados["ARIMA(1, 0, 0)"]
    pred_test = np.asarray(modelo_seleccionado.forecast(steps=len(test)), dtype=float)
    mae = mean_absolute_error(test.values, pred_test)
    mse = mean_squared_error(test.values, pred_test)
    rmse = np.sqrt(mse)

    c1, c2, c3 = st.columns(3)
    c1.metric("Modelo seleccionado", "ARIMA(1,0,0)")
    c2.metric("MAE en prueba", f"{mae:.4f} días")
    c3.metric("RMSE en prueba", f"{rmse:.4f} días")

    with st.expander("Comparar todos los modelos evaluados", expanded=True):
        st.dataframe(
            tabla_modelos.style.format({"MAE": "{:.4f}", "MSE": "{:.4f}", "RMSE": "{:.4f}"}),
            use_container_width=True, hide_index=True
        )
        st.caption("Un MAE o RMSE menor indica menor error en el periodo de prueba. La comparación se realiza sobre los mismos ocho meses.")

    # Real vs predicho en el conjunto de prueba
    pred_df = pd.DataFrame({
        "Mes": test.index,
        "Valor real": test.values,
        "Pronóstico ARIMA(1,0,0)": pred_test
    })
    pred_df["Error absoluto"] = np.abs(pred_df["Valor real"] - pred_df["Pronóstico ARIMA(1,0,0)"])

    if plotly_disponible:
        fig_test = go.Figure()
        fig_test.add_trace(go.Scatter(x=train.index, y=train.values, mode="lines+markers", name="Entrenamiento"))
        fig_test.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines+markers", name="Valor real (prueba)"))
        fig_test.add_trace(go.Scatter(x=test.index, y=pred_test, mode="lines+markers", name="Pronóstico ARIMA(1,0,0)", line=dict(dash="dash", width=3)))
        fig_test.update_layout(
            title="Evaluación del pronóstico: valores reales frente a predichos",
            xaxis_title="Mes", yaxis_title="Variabilidad (días)",
            hovermode="x unified", margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_test, use_container_width=True)
    else:
        fig_test, ax_test = plt.subplots(figsize=(11, 4))
        ax_test.plot(train.index, train.values, marker="o", label="Entrenamiento")
        ax_test.plot(test.index, test.values, marker="o", label="Valor real (prueba)")
        ax_test.plot(test.index, pred_test, marker="o", linestyle="--", label="Pronóstico ARIMA(1,0,0)")
        ax_test.set_title("Evaluación del pronóstico: valores reales frente a predichos")
        ax_test.set_xlabel("Mes")
        ax_test.set_ylabel("Variabilidad (días)")
        ax_test.legend()
        ax_test.grid(True, alpha=0.3)
        fig_test.tight_layout()
        st.pyplot(fig_test)
        plt.close(fig_test)

    with st.expander("Detalle de los errores del conjunto de prueba"):
        st.dataframe(
            pred_df.assign(Mes=pred_df["Mes"].dt.strftime("%Y-%m")).style.format({
                "Valor real": "{:.3f}", "Pronóstico ARIMA(1,0,0)": "{:.3f}", "Error absoluto": "{:.3f}"
            }), use_container_width=True, hide_index=True
        )

    # --------------------------------------------------------
    # 4. Diagnóstico de residuos
    # --------------------------------------------------------
    st.divider()
    st.subheader("Diagnóstico de residuos")
    residuos = pd.Series(modelo_seleccionado.resid).dropna()
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(residuos, lags=[3], return_df=True)
        lb_pvalue = float(lb["lb_pvalue"].iloc[0])
        lb_stat = float(lb["lb_stat"].iloc[0])
        rc1, rc2 = st.columns(2)
        rc1.metric("Ljung–Box (lag 3)", f"{lb_stat:.4f}")
        rc2.metric("Valor p", f"{lb_pvalue:.4f}")
        if lb_pvalue > 0.05:
            st.success("Con p > 0,05, no se encuentra evidencia suficiente de autocorrelación residual hasta el rezago 3.")
        else:
            st.warning("Con p ≤ 0,05, hay evidencia de autocorrelación residual; el modelo podría no capturar toda la estructura temporal.")
    except Exception as exc:
        st.caption(f"No fue posible calcular Ljung–Box: {exc}")

    if plotly_disponible:
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=train.index, y=residuos.values[-len(train):] if len(residuos) >= len(train) else residuos.values,
                                     mode="lines+markers", name="Residuos"))
        fig_res.add_hline(y=0, line_dash="dash")
        fig_res.update_layout(title="Residuos del modelo ajustado en entrenamiento", xaxis_title="Observación", yaxis_title="Residuo", margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig_res, use_container_width=True)
    else:
        fig_res, ax_res = plt.subplots(figsize=(10, 3))
        ax_res.plot(residuos.values, marker="o")
        ax_res.axhline(0, linestyle="--")
        ax_res.set_title("Residuos del modelo ajustado en entrenamiento")
        ax_res.set_xlabel("Observación")
        ax_res.set_ylabel("Residuo")
        ax_res.grid(True, alpha=0.3)
        fig_res.tight_layout()
        st.pyplot(fig_res)
        plt.close(fig_res)

    st.caption("La prueba de Ljung–Box no demuestra que el modelo sea perfecto; solo evalúa si queda evidencia de autocorrelación en los rezagos comprobados.")

    # --------------------------------------------------------
    # 5. Pronóstico futuro con el modelo ajustado a toda la serie
    # --------------------------------------------------------
    st.divider()
    st.subheader("Pronóstico para los próximos meses")
    horizonte = st.slider("Meses a pronosticar", min_value=1, max_value=12, value=6, step=1, key="d2_horizonte")
    modelo_completo = ARIMA(serie, order=(1, 0, 0)).fit()
    forecast_obj = modelo_completo.get_forecast(steps=horizonte)
    forecast_mean = np.asarray(forecast_obj.predicted_mean, dtype=float)
    try:
        conf_int = forecast_obj.conf_int(alpha=0.05)
        lower = np.asarray(conf_int.iloc[:, 0], dtype=float)
        upper = np.asarray(conf_int.iloc[:, 1], dtype=float)
    except Exception:
        lower = np.full(horizonte, np.nan)
        upper = np.full(horizonte, np.nan)

    fechas_futuras = pd.date_range(
        start=serie.index[-1] + pd.offsets.MonthBegin(1), periods=horizonte, freq="MS"
    )
    forecast_df = pd.DataFrame({
        "Mes": fechas_futuras,
        "Variabilidad pronosticada (días)": forecast_mean,
        "Límite inferior 95%": lower,
        "Límite superior 95%": upper
    })

    f1, f2 = st.columns(2)
    f1.metric("Primer mes pronosticado", f"{forecast_mean[0]:.2f} días", fechas_futuras[0].strftime("%B %Y"))
    f2.metric("Último mes pronosticado", f"{forecast_mean[-1]:.2f} días", fechas_futuras[-1].strftime("%B %Y"))

    if plotly_disponible:
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=serie.index, y=serie.values, mode="lines+markers", name="Histórico", line=dict(width=3)))
        fig_future.add_trace(go.Scatter(x=fechas_futuras, y=upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig_future.add_trace(go.Scatter(x=fechas_futuras, y=lower, mode="lines", line=dict(width=0), fill="tonexty", name="Intervalo de confianza 95%", hoverinfo="skip"))
        fig_future.add_trace(go.Scatter(x=fechas_futuras, y=forecast_mean, mode="lines+markers", name="Pronóstico ARIMA(1,0,0)", line=dict(dash="dash", width=3)))
        fig_future.update_layout(
            title="Histórico y pronóstico de la variabilidad mensual",
            xaxis_title="Mes", yaxis_title="Variabilidad (días)",
            hovermode="x unified", margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_future, use_container_width=True)
    else:
        fig_future, ax_future = plt.subplots(figsize=(11, 4))
        ax_future.plot(serie.index, serie.values, marker="o", label="Histórico")
        ax_future.plot(fechas_futuras, forecast_mean, marker="o", linestyle="--", label="Pronóstico ARIMA(1,0,0)")
        if np.isfinite(lower).all() and np.isfinite(upper).all():
            ax_future.fill_between(fechas_futuras, lower, upper, alpha=0.2, label="Intervalo de confianza 95%")
        ax_future.set_title("Histórico y pronóstico de la variabilidad mensual")
        ax_future.set_xlabel("Mes")
        ax_future.set_ylabel("Variabilidad (días)")
        ax_future.legend()
        ax_future.grid(True, alpha=0.3)
        fig_future.tight_layout()
        st.pyplot(fig_future)
        plt.close(fig_future)

    tabla_forecast = forecast_df.copy()
    tabla_forecast["Mes"] = tabla_forecast["Mes"].dt.strftime("%Y-%m")
    st.dataframe(
        tabla_forecast.style.format({
            "Variabilidad pronosticada (días)": "{:.3f}",
            "Límite inferior 95%": "{:.3f}",
            "Límite superior 95%": "{:.3f}"
        }), use_container_width=True, hide_index=True
    )
    st.download_button(
        "Descargar pronóstico (CSV)",
        data=tabla_forecast.to_csv(index=False).encode("utf-8-sig"),
        file_name="pronostico_arima_variabilidad_olist.csv",
        mime="text/csv",
        key="d2_download_forecast"
    )

    st.warning(
        "**Interpretación y limitación:** el pronóstico estima la variabilidad mensual, no el tiempo de entrega de un pedido individual. "
        "Con solo 20 observaciones mensuales, el pronóstico debe interpretarse con cautela; además, el modelo puede no anticipar cambios bruscos como la caída observada al final de la serie."
    )

# ============================================================
# TAB 3 — DIMENSIÓN 3: Índice de sobrecosto por demoras — Clustering
# ⚠️ PENDIENTE: a cargo del integrante responsable del KPI 3
# ============================================================
with tab3:
    st.header("Dimensión 3: Índice de sobrecosto por demoras")
    st.markdown(
        "**Indicador:** Índice de sobrecosto por demoras &nbsp;|&nbsp; "
        "**Técnica:** Clustering (K-medias) &nbsp;|&nbsp; "
        "**Proceso asociado:** Gestión de costos logísticos"
    )
    # ============================================================================
# DIMENSIÓN 3 — Índice de sobrecosto por demoras (KPI 3)
# Técnica: Clustering (K-medias)
# ----------------------------------------------------------------------------
# Requisitos (agregar al inicio del archivo de la app si aún no están):
#
#   import pandas as pd
#   import numpy as np
#   import matplotlib.pyplot as plt
#   import seaborn as sns
#   import streamlit as st
#   from sklearn.preprocessing import StandardScaler
#   from sklearn.cluster import KMeans
#   from sklearn.metrics import silhouette_score
#
# Se asume que "df" ya existe en el script (el DataFrame consolidado
# "olist_consolidado_pedido" cargado al inicio de la app y compartido
# por las demás pestañas/tabs), tal como lo usan las Dimensiones 1, 2 y 4.
# ============================================================================

with tab3:
    st.header("Dimensión 3: Índice de sobrecosto por demoras")
    st.markdown(
        "**Indicador:** Índice de sobrecosto por demoras &nbsp;|&nbsp; "
        "**Técnica:** Clustering (K-medias) &nbsp;|&nbsp; "
        "**Proceso asociado:** Gestión de costos logísticos"
    )

    # ------------------------------------------------------------------
    # 0. Preparación de datos
    # ------------------------------------------------------------------
    df3 = df[
        (df["order_status"] == "delivered")
        & (df["order_delivered_customer_date"].notna())
    ].copy()

    # ------------------------------------------------------------------
    # 1. Cálculo del índice de sobrecosto (flete tardío vs. a tiempo)
    # ------------------------------------------------------------------
    st.subheader("1. Índice de sobrecosto por demoras")

    flete_a_tiempo = df3.loc[df3["late_delivery"] == 0, "freight_value"].mean()
    flete_tardio = df3.loc[df3["late_delivery"] == 1, "freight_value"].mean()
    sobrecosto_global = (flete_tardio - flete_a_tiempo) / flete_a_tiempo * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Flete promedio – a tiempo", f"R$ {flete_a_tiempo:.2f}")
    c2.metric("Flete promedio – tardío", f"R$ {flete_tardio:.2f}")
    c3.metric("Sobrecosto por demora", f"{sobrecosto_global:.2f}%")

    # Índice de sobrecosto a nivel de pedido: exceso del flete pagado respecto
    # al flete promedio "a tiempo" de su propio mercado (seller_state). Un
    # valor positivo indica que el pedido costó más de flete de lo esperado
    # para ese mercado; sirve como insumo para el clustering por región.
    benchmark = df3[df3["late_delivery"] == 0].groupby("seller_state")["freight_value"].mean()
    df3["freight_benchmark"] = df3["seller_state"].map(benchmark)
    df3["indice_sobrecosto"] = (
        (df3["freight_value"] - df3["freight_benchmark"]) / df3["freight_benchmark"] * 100
    )

    fig1, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(df3["indice_sobrecosto"].clip(-100, 200), bins=50, edgecolor="black")
    axes[0].set_title("Distribución del índice de sobrecosto")
    axes[0].set_xlabel("Índice de sobrecosto (%)")
    axes[0].set_ylabel("Cantidad de pedidos")
    axes[0].grid(axis="y", alpha=0.3)

    sns.boxplot(
        x=df3["late_delivery"].map({0: "A tiempo", 1: "Tardío"}),
        y=df3["freight_value"].clip(upper=150),
        ax=axes[1],
    )
    axes[1].set_title("Costo de flete: a tiempo vs. tardío")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Costo de flete (R$)")

    plt.tight_layout()
    st.pyplot(fig1)

    # ------------------------------------------------------------------
    # 2. Variables por mercado / región / vendedor (seller_state)
    # ------------------------------------------------------------------
    st.subheader("2. Variables agregadas por mercado (seller_state)")

    agg = (
        df3.groupby("seller_state")
        .agg(
            n_pedidos=("order_id", "count"),
            pct_tardios=("late_delivery", "mean"),
            freight_promedio=("freight_value", "mean"),
            price_promedio=("price", "mean"),
            indice_sobrecosto_promedio=("indice_sobrecosto", "mean"),
        )
        .reset_index()
    )
    agg["pct_tardios"] = agg["pct_tardios"] * 100
    # Se excluyen mercados con muestra muy pequeña (< 10 pedidos) por ser poco representativos
    agg = agg[agg["n_pedidos"] >= 10].reset_index(drop=True)

    st.dataframe(
        agg.sort_values("indice_sobrecosto_promedio", ascending=False).style.format(
            {
                "pct_tardios": "{:.2f}%",
                "freight_promedio": "R$ {:.2f}",
                "price_promedio": "R$ {:.2f}",
                "indice_sobrecosto_promedio": "{:.2f}%",
            }
        )
    )
    st.caption(
        f"Se analizan {agg.shape[0]} mercados (seller_state) con al menos 10 pedidos, "
        f"a partir de {len(df3):,} pedidos entregados."
    )

    # ------------------------------------------------------------------
    # 3. Clustering K-medias
    # ------------------------------------------------------------------
    st.subheader("3. Segmentación con K-medias")

    features = ["pct_tardios", "freight_promedio", "indice_sobrecosto_promedio", "price_promedio"]
    X = agg[features]
    X_scaled = StandardScaler().fit_transform(X)

    # Selección de k mediante coeficiente de silueta (k entre 2 y 5),
    # criterio recomendado cuando no existe una etiqueta objetivo previa.
    resultados_k = []
    for k in range(2, 6):
        km_k = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        resultados_k.append((k, silhouette_score(X_scaled, km_k.labels_)))
    mejor_k, mejor_sil = max(resultados_k, key=lambda t: t[1])

    kmeans = KMeans(n_clusters=mejor_k, random_state=42, n_init=10)
    agg["cluster"] = kmeans.fit_predict(X_scaled)

    st.markdown(
        f"Se evaluó *k* entre 2 y 5 mediante el coeficiente de silueta, seleccionando "
        f"**k = {mejor_k}** por presentar el mayor valor (silueta = {mejor_sil:.3f})."
    )

    fig_k, ax_k = plt.subplots(figsize=(5, 3.5))
    ks, sils = zip(*resultados_k)
    ax_k.plot(ks, sils, marker="o")
    ax_k.axvline(mejor_k, color="red", linestyle="--", alpha=0.6)
    ax_k.set_xlabel("Número de clusters (k)")
    ax_k.set_ylabel("Coeficiente de silueta")
    ax_k.set_title("Selección de k")
    ax_k.grid(alpha=0.3)
    st.pyplot(fig_k)

    # ------------------------------------------------------------------
    # 4. Gráfico de dispersión de clusters + tabla resumen
    # ------------------------------------------------------------------
    st.subheader("4. Clusters de mercados y tabla resumen")

    fig2, ax2 = plt.subplots(figsize=(7.5, 5.5))
    scatter = ax2.scatter(
        agg["pct_tardios"],
        agg["indice_sobrecosto_promedio"],
        c=agg["cluster"],
        cmap="viridis",
        s=agg["n_pedidos"] / agg["n_pedidos"].max() * 500 + 60,
        edgecolor="black",
        alpha=0.85,
    )
    for _, row in agg.iterrows():
        ax2.annotate(
            row["seller_state"],
            (row["pct_tardios"], row["indice_sobrecosto_promedio"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax2.set_xlabel("% de pedidos tardíos")
    ax2.set_ylabel("Índice de sobrecosto promedio (%)")
    ax2.set_title("Clusters de mercados (seller_state) según riesgo y sobrecosto")
    ax2.grid(alpha=0.3)
    legend1 = ax2.legend(*scatter.legend_elements(), title="Cluster")
    ax2.add_artist(legend1)
    st.pyplot(fig2)

    resumen_clusters = (
        agg.groupby("cluster")
        .agg(
            n_mercados=("seller_state", "count"),
            n_pedidos_total=("n_pedidos", "sum"),
            pct_tardios_prom=("pct_tardios", "mean"),
            sobrecosto_prom=("indice_sobrecosto_promedio", "mean"),
            freight_prom=("freight_promedio", "mean"),
        )
        .reset_index()
    )
    st.dataframe(
        resumen_clusters.style.format(
            {
                "pct_tardios_prom": "{:.2f}%",
                "sobrecosto_prom": "{:.2f}%",
                "freight_prom": "R$ {:.2f}",
            }
        )
    )

    # Perfil de los clusters (centroides estandarizados) para explicar qué
    # variable predomina en cada grupo
    centros = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    centros.T.plot(kind="bar", ax=ax3)
    ax3.set_title("Perfil de los clusters (valores estandarizados)")
    ax3.set_ylabel("Valor estandarizado")
    ax3.legend(title="Cluster")
    ax3.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    st.pyplot(fig3)

    # ------------------------------------------------------------------
    # 5. Interpretación
    # ------------------------------------------------------------------
    st.subheader("5. Interpretación")

    cluster_riesgo = resumen_clusters.loc[resumen_clusters["sobrecosto_prom"].idxmax(), "cluster"]
    cluster_modelo = resumen_clusters.loc[resumen_clusters["sobrecosto_prom"].idxmin(), "cluster"]
    estados_riesgo = agg.loc[agg["cluster"] == cluster_riesgo, "seller_state"].tolist()
    estados_modelo = agg.loc[agg["cluster"] == cluster_modelo, "seller_state"].tolist()
    sobrecosto_riesgo = resumen_clusters.loc[resumen_clusters["cluster"] == cluster_riesgo, "sobrecosto_prom"].values[0]
    sobrecosto_modelo = resumen_clusters.loc[resumen_clusters["cluster"] == cluster_modelo, "sobrecosto_prom"].values[0]

    st.info(
        f"**Cluster {cluster_riesgo} — prioridad de atención:** agrupa a {', '.join(estados_riesgo)}, "
        f"con el mayor índice de sobrecosto promedio ({sobrecosto_riesgo:.2f}%). Son los mercados donde "
        "las demoras generan el mayor impacto económico en la distribución y deberían priorizarse en los "
        "planes de mejora logística (renegociación de tarifas de flete, cambio de transportista, etc.).\n\n"
        f"**Cluster {cluster_modelo} — referencia a replicar:** agrupa a {', '.join(estados_modelo)}, "
        f"con el menor sobrecosto promedio ({sobrecosto_modelo:.2f}%) y, en general, menor porcentaje de "
        "pedidos tardíos. Sus prácticas logísticas pueden tomarse como modelo para los demás clusters."
    )


# ============================================================
# TAB 4 — DIMENSIÓN 4: Brecha de satisfacción — Árbol / Random Forest
# ============================================================
with tab4:
    st.header("Dimensión 4: Brecha de satisfacción por demora")
    st.markdown(
        "**Indicador:** Brecha de satisfacción por demora &nbsp;|&nbsp; "
        "**Técnica:** Árbol de clasificación / Random Forest &nbsp;|&nbsp; "
        "**Proceso asociado:** Atención postventa y control de calidad del servicio"
    )

    rev_atiempo = df_kpi4[df_kpi4["late_delivery"] == 0]["review_score"].mean()
    rev_tarde = df_kpi4[df_kpi4["late_delivery"] == 1]["review_score"].mean()
    brecha = (rev_atiempo - rev_tarde) / rev_atiempo * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Review promedio (a tiempo)", f"{rev_atiempo:.2f} ★")
    col2.metric("Review promedio (tardío)", f"{rev_tarde:.2f} ★")
    col3.metric("Brecha de satisfacción", f"-{brecha:.2f}%")

    st.subheader("Análisis descriptivo")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.histplot(df_kpi4["review_score"], bins=5, ax=ax, color="#4C72B0")
        ax.set_title("Distribución de review_score")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        cross = pd.crosstab(df_kpi4["late_delivery"], df_kpi4["review_score"] >= 4,
                             normalize="index") * 100
        cross.columns = ["Mala (<4)", "Buena (>=4)"]
        cross.index = ["A tiempo", "Tarde"]
        cross.plot(kind="bar", stacked=True, ax=ax, color=["#C44E52", "#55A868"])
        ax.set_title("% Satisfacción según puntualidad")
        ax.set_ylabel("% de pedidos")
        plt.xticks(rotation=0)
        st.pyplot(fig)

    st.subheader("Modelo predictivo: Random Forest")

    @st.cache_resource
    def entrenar_modelo_satisfaccion(data):
        d = data.copy()
        for c in ["seller_state", "payment_type_main"]:
            d[c + "_enc"] = LabelEncoder().fit_transform(d[c].astype(str))

        features = [
            "late_delivery", "seller_state_enc", "freight_value",
            "price", "payment_value_total", "payment_type_main_enc",
        ]
        X = d[features].fillna(0)
        y = d["satisfaccion_bin"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=50,
            criterion="gini", class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        return rf, features, X_test, y_test, y_pred, d

    with st.spinner("Entrenando Random Forest..."):
        rf, feat_rf, Xte4, yte4, ypred4, d4 = entrenar_modelo_satisfaccion(df_kpi4)

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Matriz de confusión**")
        cm = confusion_matrix(yte4, ypred4)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Mala", "Buena"], yticklabels=["Mala", "Buena"], ax=ax)
        ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
        st.pyplot(fig)
    with c2:
        st.write("**Métricas del modelo**")
        st.text(classification_report(yte4, ypred4, target_names=["Mala", "Buena"]))
        st.metric("F1-score macro", f"{f1_score(yte4, ypred4, average='macro'):.2f}")

    st.write("**Importancia de variables**")
    importancias = pd.Series(rf.feature_importances_, index=feat_rf).sort_values()
    labels_es = {
        "late_delivery": "Retraso en entrega",
        "seller_state_enc": "Estado del vendedor",
        "freight_value": "Costo de flete",
        "price": "Precio del producto",
        "payment_value_total": "Monto total pagado",
        "payment_type_main_enc": "Tipo de pago",
    }
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh([labels_es[i] for i in importancias.index], importancias.values, color="#4C72B0")
    ax.set_xlabel("Importancia (Random Forest)")
    st.pyplot(fig)

    st.write("**Prueba el modelo con tus propios valores:**")
    ci1, ci2, ci3 = st.columns(3)
    late_in = ci1.selectbox("¿El pedido llegó tarde?", ["No", "Sí"])
    freight_in4 = ci2.number_input("Costo de flete (R$)", 0.0, 2000.0, 20.0, key="freight4")
    price_in4 = ci3.number_input("Precio del producto (R$)", 0.0, 20000.0, 100.0, key="price4")

    ci4, ci5 = st.columns(2)
    pay_in4 = ci4.number_input("Monto total pagado (R$)", 0.0, 20000.0, 120.0, key="pay4")
    pay_type_in = ci5.selectbox("Tipo de pago", ["credit_card", "boleto", "voucher", "debit_card"])

    if st.button("Predecir satisfacción"):
        late_val = 1 if late_in == "Sí" else 0
        entrada = pd.DataFrame([{
            "late_delivery": late_val,
            "seller_state_enc": 0,
            "freight_value": freight_in4,
            "price": price_in4,
            "payment_value_total": pay_in4,
            "payment_type_main_enc": 0,
        }])[feat_rf]
        pred = rf.predict(entrada)[0]
        proba = rf.predict_proba(entrada)[0][1]
        if pred == 1:
            st.success(f"😊 Predicción: calificación BUENA (probabilidad: {proba*100:.1f}%)")
        else:
            st.error(f"😞 Predicción: calificación MALA (probabilidad de ser buena: {proba*100:.1f}%)")

st.markdown("---")
st.caption(
    "Grupo 8 — Modelamiento Predictivo de Datos | Universidad de Lima | "
    "Fuente: Brazilian E-Commerce Public Dataset by Olist (Kaggle)"
)
