import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from productmapper import *

import streamlit as st

st.set_page_config(layout="wide")
OUTPUT_FOLDER = "resultados/"


@st.cache(show_spinner=False)
def load_data(buffer):
    try:
        return pd.read_csv(buffer)
    except Exception:
        return pd.read_excel(buffer)


@st.cache(show_spinner=False)
def load_data_corr(buffer):
    try:
        return pd.read_csv(buffer, header=None)
    except Exception:
        return pd.read_excel(buffer, header=None)


@st.cache(show_spinner=False)
def procesar_datasets(data_nuqlea, data_corr):
    dataset_nuqlea = DatasetPM(data_nuqlea)
    dataset_corralon = DatasetPM(data_corr)

    dataset_nuqlea.procesar()
    dataset_corralon.procesar()

    return dataset_nuqlea, dataset_corralon


@st.cache(show_spinner=False)
def entrenar_pm(dataset_nuqlea, dataset_corralon):
    cols_anclaje = ["unit_singular", "brand", "category", "product_name"]

    pm = ProductMapper(verbose=True)

    pm.entrenar(
        maestro_nuqlea=dataset_nuqlea,
        maestro_corralones=dataset_corralon,
        anker_cols=cols_anclaje,
    )

    pm.agregar_anclaje("presentacion",
                       elementos=["balde", "bidon", "bolsa",
                                  "bolson", "caja", "doy",
                                  "estuche", "lata", "pallet",
                                  "rollo", "sachet", "tambor", "unidad"])

    return pm


def update_state(res):
    for index_, elem in enumerate(res):
        if elem[0]:
            try:
                result = elem[1]
                presentation_id = int(result.index.values)
                st.session_state.global_result.loc[st.session_state.num_obs]["ID Presentacion"] = presentation_id
                st.session_state.global_result.loc[st.session_state.num_obs]["ID Producto"] = result["product_id"][presentation_id]
                st.session_state.global_result.loc[st.session_state.num_obs]["Marca"] = result["brand"][presentation_id]
                st.session_state.global_result.loc[st.session_state.num_obs]["Categoria"] = result["category"][presentation_id]
                if result.get("product_name"):
                    nombre_prod = result["product_name"][presentation_id]
                else:
                    nombre_prod = result["name"][presentation_id]
                st.session_state.global_result.loc[st.session_state.num_obs]["Nombre de producto"] = nombre_prod
                st.session_state.global_result.loc[st.session_state.num_obs]["indice seleccionado"] = index_
            except IndexError:
                st.session_state.global_result.loc[st.session_state.num_obs]["ID Presentacion"] = "No encontrado"


def from_series_to_horizontal_df(df):
    df = df[df != ""]
    df = df.to_frame().T
    return df

#######################################################################################################################

st.title("ProductMapper online")

st.sidebar.header("Carga de archivos")
buf_corralon = st.sidebar.file_uploader(label="Cargar archivo de corralón")
buf_nuqlea = st.sidebar.file_uploader(label="Cargar maestro de Nuqlea")


if buf_corralon is not None and buf_nuqlea is not None:

    data_corr = load_data_corr(buf_corralon)
    data_nuqlea = load_data(buf_nuqlea)

    if 'global_result' not in st.session_state:
        st.session_state['global_result'] = pd.DataFrame(columns=['ID Corralon', 'ID Presentacion', 'ID Producto',
                                                                  'Marca', 'Categoria', 'Nombre de producto',
                                                                  'indice seleccionado'])
    if 'num_obs' not in st.session_state:
        st.session_state['num_obs'] = 0

    st.session_state.global_result["ID Corralon"] = data_corr[0]  # ID corralon

    topn = st.sidebar.slider(label="Top-N", min_value=5, max_value=20, value=10)

    csv = st.session_state.global_result.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" target="_blank">Descargar archivo CSV de Salida</a>'
    with st.sidebar.beta_container():
        st.markdown(href, unsafe_allow_html=True)


    # Session result storage
    columns_result = ['ID Corralon'] + list(data_nuqlea.columns)


    # TODO: REMOVE --
    #X_corr = data_corr.drop([0], axis=1) # Drop product y presentation ID
    #X_nuqlea = data_nuqlea.drop(["presentation_id", "join_desc_catalog"], axis=1)
    # -------
    X_corr = data_corr
    X_nuqlea = data_nuqlea

    y_corr = data_corr[1]
    y_nuqlea = data_nuqlea["presentation_id"]

    # with st.spinner("Procesando datasets..."):
    dataset_nuqlea, dataset_corralon = procesar_datasets(X_nuqlea, X_corr)

    # with st.spinner("Entrenando ProductMapper..."):
    pm = entrenar_pm(dataset_nuqlea, dataset_corralon)

    col_1, col_2 = st.beta_columns([2, 1])
    st.session_state.num_obs = col_1.number_input("Seleccione observación", min_value=0, max_value=X_corr.shape[0])
    col_2.write(" ")
    col_2.success(f"Recorrido el {(st.session_state.num_obs / X_corr.shape[0]) * 100:.2f}%")

    obs = dataset_corralon.dataset_pr.iloc[st.session_state.num_obs, -1]
    st.subheader(obs)
    with st.beta_expander("Detalle de producto corralón crudo"):
        full_cols_detalle_raw = X_corr.iloc[st.session_state.num_obs, :-1].dropna()
        st.write(from_series_to_horizontal_df(full_cols_detalle_raw))

    try:
        resultado, output = pm.predecir(obs, thresh_ir=1, min_sim=1, top_n=topn)
        with st.form(key="my_form"):

            res = []
            previous_selected_index = st.session_state.global_result.loc[st.session_state.num_obs]["indice seleccionado"]

            for ix, i in enumerate(range(topn)):
                try:
                    scores = resultado.iloc[:, -1].unique()
                    score_actual = resultado.iloc[ix, -1]
                    string_to_show = "mas probable" if score_actual == scores[0] else "menos probable"
                    with st.beta_expander(f"Candidato {ix + 1} - {string_to_show} - Score {1 - score_actual:.2f}", expanded=True):
                        partial_result = resultado.iloc[ix, :-2]    # exclude MAIN_COL y SCORE
                        st.dataframe(from_series_to_horizontal_df(partial_result))
                        res.append([st.checkbox("Seleccionar", key=f"{st.session_state.num_obs}{ix}{i}",
                                                value=previous_selected_index == ix),
                                    from_series_to_horizontal_df(partial_result)])
                except IndexError:
                    break
            with st.beta_expander("", expanded=True):
                res.append([st.checkbox("Seleccionar si el resultado NO fue encontrado entre los candidatos",
                                        key=f"{st.session_state.num_obs}{ix+1}{i}", value=False)])
            submit_button = st.form_submit_button(label="Guardar", on_click=update_state(res))
            if submit_button:
                st.write("Resultado guardado")
    except ValueError:
        st.subheader("Ninguna producto parecido fue encontrado.")
    st.header("Resultado parcial")
    st.write(st.session_state.global_result)

    # TODO: Indexar maestro corralones por primera col
