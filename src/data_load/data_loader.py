"""
@File - pipeline.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 07/07/2026
"""

from pathlib import Path
import shutil
import pandas as pd
import dlt
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    ValidationError,
    model_validator,
)
from .models.trial_model import Trial
from .standardise import standardise_columns
from .config import load_config

ALWAYS_DROP_COLS = [
    # "acb",
    # "acb24",
    "action",
    "archived",
    # "average_length_mm_0_25",
    # "average_lenght_mm_0_28",
    # "average_length_mm_0_35",
    # "average_length_mm_gte25",
    # "average_length_mm_gte28",
    # "average_length_mm_gte35",
    # "average_length_mm_gte40",
    # "average_length_mm_gte45",
    # "average_length_mm_gte50",
    # "average_length_mm_gte55",
    # "average_length_mm_gte60",
    # "average_length_mm_gte65",
    # "average_length_mm_gte70",
    # "average_length_mm_gte75",
    # "average_length_mm_gte80",
    # "average_length_mm_25_28",
    # "average_length_mm_28_35",
    # "average_length_mm_35_40",
    # "average_length_mm_40_45",
    # "average_length_mm_45_50",
    # "average_length_mm_50_55",
    # "average_length_mm_55_60",
    # "average_length_mm_60_65",
    # "average_length_mm_65_70",
    # "average_length_mm_70_75",
    # "average_length_mm_75_80",
    # "average_lenght_mm_75_80",
    # "average_lenght_mm_gte75",
    # "average_lenght_mm_gte80",
    # "average_length_mm_total",
    "average_nir_value_total",
    # "audpc",
    "barcode_39",
    "boxnos",
    "block",
    "blackleg",
    "batch_id",
    "blass11",
    "blass10",
    "blass12",
    "blass1",
    "blass2",
    "blass3",
    "blass4",
    "blass5",
    "blass6",
    "blass7",
    "blass8",
    "blass9",
    # "blightcoun",
    # "blightscal",
    "bn",
    "cross_name",
    "check",
    # "crisp",
    "cleanyield",
    "cleanyeild",
    "col_0_28_hash",
    "col_0_28_percent",
    "col_0_25_hash",
    "col_0_25_percent",
    "col_25_28_hash",
    "col_25_28_percent",
    "col_28_35_hash",
    "col_28_35_percent",
    "col_35_40_hash",
    "col_35_40_percent",
    "col_40_45_hash",
    "col_40_45_percent",
    "col_45_50_hash",
    "col_45_50_percent",
    "col_50_55_hash",
    "col_50_55_percent",
    "col_55_60_hash",
    "col_55_60_percent",
    "col_60_65_hash",
    "col_60_65_percent",
    "col_65_70_hash",
    "col_65_70_percent",
    "col_70_75_hash",
    "col_70_75_percent",
    "col_0_35_hash",
    "col_0_35_percent",
    "col_75_80_percent",
    "col_75_80_hash",
    "col_75_80hash",
    "col_75_80percent",
    "com_year4",
    "crisp_yr4",
    "cross_id",
    "clone",
    "cone",
    "chpcom3m8d",
    "chpcom6m4d",
    "cook_com",
    "chpcom3m4d",
    "chpcom6m8d",
    "crisp_3m8d",
    "crisp_3m4d",
    "chip_3m8d",
    "chip_3m4d",
    "crisp_6m8d",
    "crisp_6m4d",
    "chip_6m8d",
    "chip_6m4d",
    "description1",
    "defectuk",
    "discards",
    "dutchscore",
    # "dryrot",
    # "drought",
    "datetime",
    # "dryness",
    "entry",
    # "emergence",
    "end_plot",
    "emergno",
    "female",
    # "fcover",
    # "field_plan",
    # "ff_blight",
    # "ff_sg",
    # "folgone",
    "female_pedigree",
    "for_yield",
    # "flesh",
    # "flowercolr",
    # "ffrots",
    "fn",
    "fplot",
    "fplant",
    "fol_com",
    "fol_dev",
    "fol_height",
    "fcover",
    "folcomment",
    "filler",
    "flavour",
    "gand1",
    "gand2",
    "gand3",
    "ganw1",
    "ganw2",
    "ganw3",
    "genot",
    "green",
    "growth_cr",
    "gte75_hash",
    "gte75_percent",
    "gte80_hash",
    "gte80_percent",
    "grad_com",
    "hla_com",
    "harvester",
    "huntl_3m8d",
    "hunta_3m8d",
    "huntb_3m8d",
    "huntl_3m4d",
    "hunta_3m4d",
    "huntb_3m4d",
    "huntl_6m8d",
    "huntl_6m4d",
    "hunta_6m8d",
    "hunta_6m4d",
    "huntb_6m8d",
    "huntb_6m4d",
    # "hunterl",
    # "huntera",
    # "hunterb",
    "images",
    "identity",
    "identity1",
    "loc",
    "locn",
    "mplot",
    "mplant",
    "male",
    "mate",
    "mat",
    "mat2",
    "mat3",
    "mate2",
    "mate3",
    "mat_yr4",
    "matur1",
    "male_pedigree",
    "max_average_nir",
    "maxhash_average_nir",
    "method",
    # "mealiness",
    "mech",
    "min_average_nir",
    "minhash_average_nir",
    "miss",
    "mknos",
    "namesg",
    "original_name",
    "others",
    "over40no",
    "over80nos",
    "over_40",
    "over_70",
    "over_70no",
    "over_75",
    "over_75no",
    "over_80",
    "owg_g_total",
    "owg_g_5kg_total",
    "over_85",
    "over_85no",
    "overfol",
    "ovall_cook",
    "pass",
    "plot_size",
    "pallida",
    "planter",
    "plot_range",
    "plot_size_lt",
    "plot_size_uk",
    "plot_column",
    "pollination_id",
    "previous_nursery",
    "purpose",
    # "powdery",
    "pvy",
    "plot_size",
    # "rouges",
    "selection",
    "seedlot",
    "senescence",
    "skinfinco",
    "slug",
    "secgr",
    "skin",
    "scab",
    "sg_comment",
    "stor_appr",
    "stor_com",
    "stor_dorm",
    "stor_ffdam",
    "stor_ffdis",
    "stor_ffspr",
    "stor_firm",
    "stor_lspr",
    "stor_stor",
    "sprouts",
    "tyduk",
    "total",
    "total_hash",
    "total_defe",
    "totalno",
    "totperha",
    "trvind",
    "trvpercent",
    "trvsever",
    "trvweight",
    "tshp_st",
    "tuber",
    "u45tha",
    "uk45_85tha",
    # "under25",
    # "under35",
    "under25no",
    "under45no",
    "ukmy45_85",
    "verticilli",
    "virus",
    "virus_y",
    # "wet_rot",
    "waste_t_ha",
    # "weight_g_0_28",
    # "weight_g_0_25",
    # "weight_g_25_28",
    # "weight_g_28_35",
    # "weight_g_35_40",
    # "weight_g_40_45",
    # "weight_g_45_50",
    # "weight_g_50_55",
    # "weight_g_55_60",
    # "weight_g_60_65",
    # "weight_g_65_70",
    # "weight_g_70_75",
    # "weight_g_total",
    # "weight_g_0_35",
    # "weight_percent_0_35",
    # "weight_g_75_80",
    # "weight_percent_75_80",
    # "weight_percent_0_28",
    # "weight_percent_gte75",
    # "weight_percent_gte80",
    # "weight_percent_0_25",
    # "weight_percent_25_28",
    # "weight_percent_28_35",
    # "weight_percent_35_40",
    # "weight_percent_40_45",
    # "weight_percent_45_50",
    # "weight_percent_50_55",
    # "weight_percent_55_60",
    # "weight_percent_60_65",
    # "weight_percent_65_70",
    # "weight_percent_70_75",
    # "weight_percent_75_80",
    # "weight_g_gte75",
    # "weight_g_gte80",
    # "weight_g_75_80",
    "weight_g_35_50_france",
    "weight_g_50_60_france",
    "weight_g_60_75_france",
    # "weight_g_lt28_45mm",
    "yearfact",
    # "yield_gte80_t_ha",
    # "yield_0_35_t_ha",
    "yield_35_50_t_ha_france",
    "yield_50_60_t_ha_france",
    "yield_60_75_t_ha_france",
    "yield_75_t_ha_france",
    "yield_0_35_t_ha_france",
    "yield_total_t_ha_france",
    # "yieldtha",
    "yld_blight",
    "yld_green",
    # "yld_45",
    # "yld_85",
    # "y25_40",
    # "y45_65",
    # "y45_60",
    # "y65_85",
    # "y60_70",
    # "y65_75",
    # "y60_75",
    # "y35_50",
    # "y70_80",
    # "y75_85",
    "y25_40no",
    "y45_65no",
    "y80_85tha",
    "y65_75no",
    "y75_85n",
    "y60_80tha",
    "y50_60tha",
    "y45_50tha",
    "y_blightha",
    "y_defectiv",
    "y_internal",
    "y_scab",
    "ydef_tha",
    "ydis_tha",
    "ydisease",
    "ygreentha",
]

COLUMN_NAME_ALIASES: dict[str, str] = {}


def _format_errors(exc: ValidationError) -> str:
    return " | ".join(
        f'{".".join(str(x) for x in err["loc"])}: {err["msg"]}, ({err.get("input")!r})'
        for err in exc.errors()
    )


def discover_excel_files(source_dir: str, file_extensions: list[str]) -> list[Path]:
    incoming_dir = Path(source_dir)
    extensions = {e.lower() for e in file_extensions}
    files = [
        f
        for f in incoming_dir.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]
    return sorted(files)


def archive_loaded_sheet(archive_dir: Path, file: Path):
    target_path = archive_dir / file.name

    if target_path.exists():
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        target_path = archive_dir / f"{file.stem}_{timestamp}{file.suffix}"

    shutil.move(str(file), str(target_path))


@dlt.resource(name="raw_excel_rows", write_disposition="skip")
def load_excel_source(file_path: str, sheet_name: str | int):
    file_name = Path(file_path).name
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    std_result = standardise_columns(df, ALWAYS_DROP_COLS, COLUMN_NAME_ALIASES)
    df = std_result.dataframe

    valids = []
    rejects = []

    for idx, row in enumerate(df.to_dict(orient="records"), start=2):
        row["source_file"] = file_name
        row["source_row"] = idx

        try:
            validated_row = Trial(**row).model_dump()
            valids.append(dict(validated_row))
        except ValidationError as err:
            reject_row = dict(row)
            reject_row["error_message"] = _format_errors(err)
            rejects.append(reject_row)
            print(reject_row)

    yield {"valids": valids, "rejects": rejects}


@dlt.transformer(
    data_from=load_excel_source,
    name="valid",
    write_disposition={"disposition": "merge", "strategy": "scd2"},
    merge_key=[
        "year",
        "location",
        "experiment_name",
        "name1",
        "plot",
    ],
    schema_contract={
        "tables": "evolve",
        "columns": "evolve",
        "data_type": "freeze",
    },
)
def process_valid_rows(excel_rows):
    yield from excel_rows["valids"]


@dlt.transformer(
    data_from=load_excel_source,
    name="quarantine",
    write_disposition={"disposition": "append"},
)
def process_quarantine_rows(excel_rows):
    yield from excel_rows["rejects"]


def main():
    config = load_config()
    files = discover_excel_files(config.paths.incoming, config.excel.extensions)

    if not files:
        print(
            f"No file to process. Put excel files in the incoming folder. Allowed file extensions: {config.excel.extensions}"
        )
        return

    db_dir = Path(config.paths.duckdb)

    archive_dir = Path(config.paths.archive)
    archive_dir.mkdir(parents=True, exist_ok=True)

    pipeline = dlt.pipeline(
        pipeline_name="trial_data",
        destination=dlt.destinations.duckdb(destination_name=str(db_dir)),
        dataset_name="trial_data",
    )
    pipeline.drop_pending_packages()

    for file in files:
        print(f"Processing file: {file}")

        try:
            source_instance = load_excel_source(str(file), config.excel.default_sheet)

            load_info = pipeline.run(
                [
                    source_instance | process_valid_rows,
                    source_instance | process_quarantine_rows,
                ]
            )

            for package in load_info.load_packages:
                if package.schema_update:
                    print(
                        f"SCHEMA UPDATED: {file.name} : {list(package.schema_update.keys())}"
                    )

            load_failure = load_info.has_failed_jobs

            if not load_failure:
                archive_loaded_sheet(archive_dir, file)
            else:
                print(f"LOAD FAILED: {file.name}. Leaving in incoming folder.")

        except Exception as pipeline_error:
            print(
                f"CRITICAL ERROR: Failed to fully process {file.name}. Leaving in incoming folder. Error: {pipeline_error}"
            )
            pipeline.drop_pending_packages()


if __name__ == "__main__":
    main()
