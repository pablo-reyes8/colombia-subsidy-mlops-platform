import pandas as pd

from colombia_subsidy_ml.data.build import build_dataset


def _df_with_cols(cols, data):
    return pd.DataFrame(data, columns=cols)


def test_build_dataset_minimal():
    ids = [1, 2, 3]

    generales = _df_with_cols(
        ["DIRECTORIO", "DPTO", "P6040", "P6080", "P6100", "P3039", "P3042"],
        [
            [1, 11, 30, 1, 1, 1, 3],
            [2, 11, 40, 2, 2, 2, 4],
            [3, 11, 50, 1, 1, 1, 2],
        ],
    )

    laborales = _df_with_cols(
        [
            "DIRECTORIO",
            "DPTO",
            "P6460",
            "P6440",
            "P6426",
            "INGLABO",
            "P6620S1",
            "P6510S1",
            "P6585S1A1",
            "P6585S2A1",
            "P6585S3A1",
            "P6585S4A1",
            "P6545S1",
            "P6580S1",
            "P6630S1A1",
            "P6630S2A1",
            "P6630S3A1",
            "P6630S4A1",
            "P6800",
            "P6920",
            "P7140S7",
            "P514",
            "P7140S6",
            "P7100",
        ],
        [
            [1, 11, 1, 1, 12, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 1, 0, 1, 0, 0],
            [2, 11, 2, 2, 8, 2000, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 35, 1, 0, 1, 0, 0],
            [3, 11, 1, 1, 6, 1500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 1, 0, 1, 0, 0],
        ],
    )

    hogar = _df_with_cols(
        [
            "P4030S1",
            "P4020",
            "P4030S3",
            "P4030S5",
            "P6008",
            "DIRECTORIO",
            "P5090",
            "P5100",
            "P5140",
            "P4030S2",
            "P4030S4",
        ],
        [
            [1, 1, 1, 1, 4, 1, 1, 200, 300, 1, 1],
            [1, 1, 1, 1, 3, 2, 1, 100, 200, 1, 1],
            [1, 1, 1, 1, 2, 3, 1, 0, 150, 1, 1],
        ],
    )

    subsidios = _df_with_cols(
        [
            "P7510S3",
            "DIRECTORIO",
            "P7510S2A1",
            "P7510S3A1",
            "P750S1A1",
            "P750S2A1",
            "P1661S1A1",
            "P1661S2A1",
            "P1661S3A1",
        ],
        [
            [1, 1, 100, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 0],
        ],
    )

    fuerza_trabajo = _df_with_cols(["P6240", "DIRECTORIO"], [[1, 1], [3, 2], [2, 3]])

    desempleados = _df_with_cols(
        ["P7250", "P7440S1", "P9460", "P7422", "DIRECTORIO"],
        [[1, 1, 0, 0, 1], [2, 2, 0, 0, 2], [3, 3, 0, 0, 3]],
    )

    df = build_dataset(
        {
            "generales": generales,
            "laborales": laborales,
            "hogar": hogar,
            "subsidios": subsidios,
            "fuerza_trabajo": fuerza_trabajo,
            "desempleados": desempleados,
        }
    )

    assert len(df) == 3
    assert "Subsidio" in df.columns
    assert df["Subsidio"].isna().sum() == 0
    assert df["Subsidio"].dtype.kind in {"i", "u"}
    assert "CLASE_SOCIAL" in df.columns
    assert "precio_combinado" in df.columns
    assert "Numero_personas" in df.columns
