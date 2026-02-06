from __future__ import annotations

from functools import reduce
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _select_rename(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    cols = list(mapping.keys())
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    return df[cols].rename(columns=mapping)


def _prepare_generales(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "DIRECTORIO": "Directorio",
        "DPTO": "Departamento",
        "P6040": "Edad",
        "P6080": "Etnia",
        "P6100": "Sistema de Salud",
        "P3039": "Genero",
        "P3042": "Educacion Maxima",
    }
    return _select_rename(df, mapping)


def _prepare_laborales(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "DIRECTORIO": "Directorio",
        "DPTO": "Departamento",
        "P6440": "Tiene contrato",
        "P6460": "Tipo de contrato",
        "P6426": "Meses en la empresa",
        "INGLABO": "INGRESOS BASE",
        "P6620S1": "Ingresos especie",
        "P6510S1": "Ingresos extra",
        "P6585S1A1": "Auxilio alimentacion",
        "P6585S2A1": "Auxilio Transporte",
        "P6585S3A1": "Subsidio Familiar",
        "P6585S4A1": "Subsidio Educativo",
        "P6545S1": "Primas",
        "P6580S1": "Bonificaciones",
        "P6630S1A1": "Prima servicios",
        "P6630S2A1": "Prima navidad",
        "P6630S3A1": "Prima vacaciones",
        "P6630S4A1": "Primas anuales",
        "P6800": "Horas de trabajo semanales",
        "P6920": "Fondo de pensiones",
        "P7140S7": "Exigencia fisica o mental",
        "P514": "Trabajo estable",
        "P7140S6": "Desagrado por el trabajo",
        "P7100": "Horas Adicionales",
    }

    df = _select_rename(df, mapping)

    ingresos_cols = [
        "Ingresos especie",
        "Ingresos extra",
        "Auxilio alimentacion",
        "Auxilio Transporte",
        "Subsidio Familiar",
        "Subsidio Educativo",
        "Primas",
        "Bonificaciones",
        "Prima servicios",
        "Prima navidad",
        "Prima vacaciones",
        "Primas anuales",
    ]
    df[ingresos_cols] = df[ingresos_cols].fillna(0)
    df["Ingresos finales"] = df["INGRESOS BASE"] + df[ingresos_cols].sum(axis=1)

    keep = [
        "Directorio",
        "Tipo de contrato",
        "Tiene contrato",
        "Meses en la empresa",
        "Fondo de pensiones",
        "Exigencia fisica o mental",
        "Trabajo estable",
        "Desagrado por el trabajo",
        "Horas Adicionales",
        "Ingresos finales",
        "Subsidio Familiar",
        "Subsidio Educativo",
    ]
    return df[keep]


def _prepare_fuerza_trabajo(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "DIRECTORIO": "Directorio",
        "P6240": "Actividad realizada",
    }
    df = _select_rename(df, mapping)
    df["Empleado"] = df["Actividad realizada"].isin([1, 3]).astype(int)
    return df[["Directorio", "Actividad realizada", "Empleado"]]


def _prepare_desempleados(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "DIRECTORIO": "Directorio",
        "P7250": "Semanas Buscando Trabajo",
        "P7440S1": "Meses sin trabajo",
        "P7422": "Ingresos Mes Pasado",
        "P9460": "Subsidio Desempleo",
    }
    return _select_rename(df, mapping)


def _prepare_hogar(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "DIRECTORIO": "Directorio",
        "P6008": "N_Personas",
        "P5090": "Tipo de Vivienda",
        "P5100": "Cuota Amotrtizacion",
        "P5140": "Arriendo",
        "P4030S1": "Energia Electrica",
        "P4020": "Material Casa",
        "P4030S3": "Alcantarillado",
        "P4030S5": "Acueducto",
        "P4030S2": "Gas Natural",
        "P4030S4": "Recoleccion Basura",
    }
    return _select_rename(df, mapping)


def _prepare_subsidios(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "DIRECTORIO": "Directorio",
        "P7510S3": "Subsidios",
        "P7510S2A1": "Valor 1",
        "P7510S3A1": "Valor 2",
        "P750S1A1": "Valor 3",
        "P750S2A1": "Valor 4",
        "P1661S1A1": "Valor 5",
        "P1661S2A1": "Valor 6",
        "P1661S3A1": "Valor 7",
    }
    df = _select_rename(df, mapping)
    value_cols = [c for c in df.columns if c.startswith("Valor ")]
    df[value_cols] = df[value_cols].fillna(0)
    df["Valor Subsidio"] = df[value_cols].sum(axis=1)
    df = df.drop(columns=value_cols)
    return df


def _combinar_valores(row: pd.Series) -> float:
    a = row.get("Arriendo")
    b = row.get("Cuota Amotrtizacion")
    if pd.isna(a) and pd.isna(b):
        return np.nan
    if pd.isna(a):
        return b
    if pd.isna(b):
        return a
    return (a + b) / 2


def build_dataset(
    raw_tables: Dict[str, pd.DataFrame],
    *,
    keep_missing_target: bool = False,
) -> pd.DataFrame:
    """Build consolidated modeling dataset from raw GEIH tables."""
    generales = _prepare_generales(raw_tables["generales"])
    laborales = _prepare_laborales(raw_tables["laborales"])
    hogar = _prepare_hogar(raw_tables["hogar"])
    subsidios = _prepare_subsidios(raw_tables["subsidios"])
    fuerza_trabajo = _prepare_fuerza_trabajo(raw_tables["fuerza_trabajo"])
    desempleados = _prepare_desempleados(raw_tables["desempleados"])

    dataframes = [generales, laborales, hogar, fuerza_trabajo, desempleados, subsidios]
    df_final = reduce(lambda left, right: pd.merge(left, right, on="Directorio", how="inner"), dataframes)

    df_final["Costo Vivienda"] = df_final.apply(_combinar_valores, axis=1)

    target_sources = ["Subsidio Familiar", "Subsidio Educativo", "Subsidio Desempleo", "Subsidios"]
    df_final["Subsidio"] = np.where(
        df_final[target_sources].isna().all(axis=1),
        np.nan,
        np.where(
            (df_final["Subsidio Desempleo"] == 1) | (df_final["Subsidios"] == 1),
            1,
            0,
        ),
    )

    df_final["Sistema de Salud"] = df_final["Sistema de Salud"].fillna(0)
    df_final["Tipo de contrato"] = df_final["Tipo de contrato"].fillna(0)
    df_final.loc[(df_final["Tiene contrato"] == 2), "Tipo de contrato"] = 0
    df_final["Horas Adicionales"] = df_final["Horas Adicionales"].fillna(0)
    df_final["Meses sin trabajo"] = df_final["Meses sin trabajo"].fillna(0)

    intervalos = [0, 420000, 1680000.0, 8400000.0, float("inf")]
    nombres = ["Pobreza Monetaria", "Clase Baja", "Clase media", "Clase alta"]
    df_final["CLASE_SOCIAL"] = pd.cut(
        df_final["Ingresos finales"], bins=intervalos, labels=nombres, right=False
    )

    drop_cols = [
        "Genero",
        "Subsidio Educativo",
        "Subsidio Desempleo",
        "Subsidio Familiar",
        "Subsidios",
        "Semanas Buscando Trabajo",
    ]
    df_final = df_final.drop(columns=[c for c in drop_cols if c in df_final.columns])

    if not keep_missing_target:
        df_final = df_final[df_final["Subsidio"].notna()].copy()
        df_final["Subsidio"] = df_final["Subsidio"].astype(int)

    return df_final


def load_raw_tables(raw_dir: str | Path, *, encoding: str, sep: str, raw_tables: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    tables = {}
    for key, filename in raw_tables.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing raw table: {path}")
        tables[key] = pd.read_csv(path, sep=sep, encoding=encoding)
    return tables


def build_from_config(config: dict) -> pd.DataFrame:
    tables = load_raw_tables(
        config["raw_dir"],
        encoding=config.get("encoding", "latin-1"),
        sep=config.get("sep", ";"),
        raw_tables=config["raw_tables"],
    )
    return build_dataset(tables)


def save_dataset(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
