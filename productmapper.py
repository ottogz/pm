
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
import copy


# Helper funcs

def eliminar_acentos(string):
    """
    DESCRIPCION: Eliminar acentos de una cadena de texto
    PARAMS:
    * string (string): Cadena de texto
    RETURN:
    * Cadena de texto sin acentos
    """
    string = str(string)
    trans = str.maketrans('ÁÉÍÓÚÜáéíóúü', 'AEIOUUaeiouu')
    return string.translate(trans)


def process_str_col(serie, get_unique=False):
    """
    DESCRIPCION: Procesar una columna de dataframe como string.
    PARAMS:
    * serie (pd.serie): Columna para procesar.
    RETURN:
    * Columna procesada.
    """

    # Lower
    serie = serie.str.lower()

    # Eliminar acentos
    serie = serie.apply(lambda x: eliminar_acentos(x))

    # Reemplazar ñ por n
    serie = serie.apply(lambda x: x.replace("ñ", "n"))

    # Eliminar símbolos
    serie = serie.apply(lambda x: re.sub("[^a-zA-Z0-9]", " ", x))

    # Eliminar nans
    serie = serie.apply(lambda x: " ".join([str(i) for i in x.split() if str(i) != "nan"]))

    # Striping
    serie = serie.str.strip()
    serie = serie.apply(lambda x: " ".join(x.split()))

    # Eliminar ""

    if get_unique:

        temp = []
        serie.apply(lambda x: temp.append(x) if x != "" else ...)
        unique = list(set(temp))

        return unique

    else:
        return serie


def distext(token, candidato_ancla):
    score = 0
    for ix, i in enumerate(token):
        try:
            if i == candidato_ancla[ix]:
                score += 1
            else:
                score -= 1
        except:
            score -= 1

    return score / len(candidato_ancla)


class DatasetPM:

    def __init__(self, dataset):

        self.dataset_raw = dataset
        self.dataset_pr = None
        self.normalize_cols = None
        self.main_col = None

    def normalizar(self, normalize_cols):
        self.normalize_cols = normalize_cols
        self.dataset_pr = copy.deepcopy(self.dataset_raw)
        if normalize_cols == "all":
            for columna in self.dataset_pr.columns:
                self.dataset_pr[columna] = process_str_col(self.dataset_pr[columna].astype(str))
        elif isinstance(normalize_cols, list):
            for columna in normalize_cols:
                self.dataset_pr[columna] = process_str_col(self.dataset_pr[columna].astype(str))
        elif normalize_cols == None:
            pass
        else:
            raise ValueError(f"No se entiende la instrucción '{normalize_cols}'")

    def joinear(self, join_cols):
        self.join_cols = join_cols
        if join_cols == "all":
            self.dataset_pr["MAIN_COL"] = self.dataset_pr.apply(lambda x: " ".join([i for i in x]), axis=1)
        elif isinstance(join_cols, list):
            self.dataset_pr["MAIN_COL"] = self.dataset_pr.loc[:, join_cols].apply(lambda x: " ".join([i for i in x]),
                                                                                  axis=1)
        elif join_cols == None:
            pass
        else:
            raise ValueError(f"No se entiende la instrucción '{join_cols}'")

    def procesar(self, normalize_cols="all", join_cols="all"):
        self.normalizar(normalize_cols)
        self.joinear(join_cols)


class ProductMapper:

    def __init__(self, verbose=False):

        self.verbose = verbose
        self.anker_cols = None
        self.predict_cols = None
        self.master_ankers = None
        self.maestro_nuqlea = None
        self.maestro_corralones = None
        self.desc_col = None
        self.df_vecs_nuqlea = None
        self.df_vecs_corralon = None

    def entrenar(self, maestro_nuqlea, maestro_corralones, anker_cols):

        self.maestro_nuqlea = maestro_nuqlea.dataset_pr.copy()
        self.maestro_corralones = maestro_corralones.dataset_pr.copy()
        self.anker_cols = anker_cols

        # Paso 1 - Extraemos las anclas

        master_ankers = {}

        for col in tqdm(anker_cols,
                        desc="Extrayendo master ankers...",
                        disable=not self.verbose,
                        ncols=75):

            # Si el punto de anclaje está en la tabla, extraemos sus valores únicos
            if col in self.maestro_nuqlea.columns:
                master_ankers[col] = process_str_col(self.maestro_nuqlea[col],
                                                     get_unique=True)
            # Si no, printeamos warning
            else:
                print(f"Warning: El ancla {col} no ingresa al master_ankers porque no está en el dataset")

        # Guardamos el master_ankers
        self.master_ankers = master_ankers

        # Paso 2 - Vectorizar los dos maestros

        vectorizer_nuqlea = TfidfVectorizer(min_df=2, max_features=1500)
        vectorizer_corralon = TfidfVectorizer(min_df=2, max_features=1500)

        vecs_raw_nuqlea = vectorizer_nuqlea.fit_transform(self.maestro_nuqlea["MAIN_COL"])
        vecs_raw_corralon = vectorizer_corralon.fit_transform(self.maestro_corralones["MAIN_COL"])

        self.df_vecs_nuqlea = pd.DataFrame(vecs_raw_nuqlea.todense(),
                                           columns=vectorizer_nuqlea.get_feature_names())

        self.df_vecs_corralon = pd.DataFrame(vecs_raw_corralon.todense(),
                                             columns=vectorizer_corralon.get_feature_names())

    def agregar_anclaje(self, nombre, elementos):

        """
        DESCRIPCION: Método para agregar un anclaje que no se encuentre en las columnas
            del maestro de nuqlea.
        """

        if self.master_ankers is None:
            raise ValueError("El ProductMapper todavía no fue entrenado!")

        self.master_ankers[nombre] = elementos

        if nombre in self.maestro_nuqlea.columns:
            self.maestro_nuqlea[nombre] = process_str_col(self.maestro_nuqlea[nombre])
        else:
            if self.verbose:
                print(f"Warning: El ancla '{nombre}' no se encuentra en el maestro de nuqlea")

        if self.verbose:
            print(f"El punto de anclaje '{nombre}' se agregó correctamente!")

    def get_anclas(self, target_obs, min_sim=10):

        output = {}

        # Recorremos todos los puntos de anclaje
        for anker in self.master_ankers:

            # Generamos dos acumuladores, uno para distancias y
            # otro para las opciones del punto de anclaje
            candidatos_dist = np.array([])
            candidatos_elem = np.array([])

            # Para cada opción del punto de anclaje...
            for anker_opt in self.master_ankers.get(anker):

                # si el ancla es compuesta
                #   Toma el ancla solo si todos las palabras aparecen en orden en la observacion con min_sim
                if len(anker_opt.split(" ")) > 1:

                    is_in = 0
                    scores = []

                    for ix, token in enumerate(target_obs.split()):

                        dist = distext(token, anker_opt.split()[0])

                        if dist >= min_sim:

                            is_in += 1
                            scores.append(dist)

                            for i in range(len(anker_opt.split(" ")) - 1):
                                try:
                                    dist = distext(target_obs.split()[ix + i + 1], anker_opt.split(" ")[i + 1])
                                except Exception:
                                    break

                                if dist >= min_sim:
                                    is_in += 1
                                    scores.append(dist)

                            if len(anker_opt.split(" ")) == is_in:
                                candidatos_dist = np.append(candidatos_dist, np.mean(scores))
                                candidatos_elem = np.append(candidatos_elem, anker_opt)

                # Calculamos su distancia de cada token de la observación
                for token in target_obs.split():

                    dist = distext(token, anker_opt)

                    # Si la distancia es mayor al threshold y la opción todavía no se agregó
                    # al array de candidatos, lo agregamos y a su distancia.
                    if dist >= min_sim and anker_opt not in candidatos_elem:
                        candidatos_dist = np.append(candidatos_dist, dist)
                        candidatos_elem = np.append(candidatos_elem, anker_opt)

            # Una vez que recorrimos todas las opciones del punto de anclaje, consolidamos la info

            if len(candidatos_elem) > 0:
                output[anker] = candidatos_elem[np.argsort(candidatos_dist)[-5:]][::-1]
            else:
                output[anker] = None

        return output

    def information_retrieval_(self, subset, target_obs, vectorizer_args, thresh=0.2, top_n=5):

        vectorizer = TfidfVectorizer(*vectorizer_args)
        vecs_master = vectorizer.fit_transform(subset["MAIN_COL"])
        vec_obs = vectorizer.transform([target_obs])
        distancias = pairwise_distances(vecs_master.todense(),
                                        vec_obs.todense(),
                                        metric="cosine")

        self.vecs_master = vecs_master
        self.vec_obs = vec_obs

        if np.max(distancias) >= thresh:

            subset["METRIC"] = distancias
            output = subset[subset["METRIC"] >= thresh].sort_values("METRIC", ascending=False)

            return output.iloc[:top_n, :]

        else:
            print(f"No pudimos encontrar ningún match lo suficientemente similar. Modificá el parámetro 'thresh' considerando que el max es {np.max(distancias):.2f}")

    def information_retrieval(self, subset, target_obs, vectorizer_args, anclas, thresh=0.2, top_n=5):

        for i in anclas:
            an_ = anclas.get(i)
            if an_ is not None:
                target_obs += f" {an_[0]}"

        #         print(target_obs)

        vectorizer = CountVectorizer(*vectorizer_args)
        vecs_master = vectorizer.fit_transform([target_obs])
        vec_obs = vectorizer.transform(subset["MAIN_COL"])
        distancias = pairwise_distances((vecs_master.todense() > 0).astype(int),
                                        (vec_obs.todense() > 0).astype(int),
                                        metric="hamming")

        if np.min(distancias) <= thresh:
            # Si hay alguna de las distancias que es menor al threshold...
            subset.loc[:, "METRIC"] = distancias.flatten()
            output = subset[subset.loc[:, "METRIC"] <= thresh].sort_values("METRIC", ascending=True)

            return output.iloc[:top_n, :]

        else:
            print(
                f"No pudimos encontrar ningún match lo suficientemente similar. Modificá el parámetro 'thresh' considerando que el max es {np.max(distancias):.2f}")

    def buscador(self, serie, buscar):

        """
        DESCRIPCION: Función interna para facilitar el filtrado del maestro.
        """

        return serie.apply(lambda x: buscar in x)

    def predecir(self, target_obs, min_sim=1,
                 top_n=5, thresh_ir=0.2, vectorizer_args={"ngram_range": (1, 1),
                                                          "min_df": 2}):

        """
        DESCRIPCION: Método para identificar el match más probable tomando una observación del maestro
            de los corralones. Dicha observación debe ser una columna que concatene los valores de las
            columnas de los puntos de anclaje.
        """

        if self.master_ankers is None:
            raise ValueError("El ProductMapper todavía no fue entrenado!")

        output = self.get_anclas(target_obs, min_sim=min_sim)
        mask = []
        for ancla in output:
            temp_mask = []
            if output.get(ancla) is not None:
                for candidato in output.get(ancla):
                    if candidato is not None:
                        temp_mask.append(
                            self.buscador(self.maestro_nuqlea["MAIN_COL"], candidato).values.reshape(-1, 1))

                temp_mask = np.concatenate(temp_mask, axis=1).any(axis=1).reshape(-1, 1)
                mask.append(temp_mask)

        mask = np.concatenate(mask, axis=1).any(axis=1)

        #         print(sum(mask)/len(mask))

        descripciones = self.maestro_nuqlea[mask]
        match = self.information_retrieval(descripciones,
                                           target_obs,
                                           vectorizer_args,
                                           anclas=output,
                                           top_n=top_n,
                                           thresh=thresh_ir)

        return match, output


