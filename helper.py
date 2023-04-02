

def rename_nui_columns(df):
    
    return df.rename(
            columns = {
                "V0": "id",
                "V1": "city",
                "V2a": "municipality",
                "V2b": "municipality_group",
                "V3a": "nui_name",
                "V3b": "nui_source",
                "V4a": "residencies_number",
                "V4b": "residencies_source",
                "V4c": "residencies_date",
                "V5a": "nui_type_category",
                "V5b": "nui_type_spec",
                "V5c": "nui_type_proportion",
                "V6": "nui_date_establishment",
                "V7": "nui_real_estate_dynamics",
                "V8": "urban_contiguity",
                "V9": "zeis",
                "V10a": "protected_areas_category",
                "V10b": "protected_areas_type",
                "V11a": "permanent_preservation",
                "V11b": "permanent_preservation_type",
                "V11c": "permanent_preservation_source",
                "V12a": "risk_situation",
                "V12b": "risk_situation_spec",
                "V13a": "risk_susceptibility",
                "V13b": "susceptibility_level",
                "V14a": "tracing",
                "V14b": "tracing_spec",
                "V15a": "lots",
                "V15b": "lots_spec",
                "V16a": "buildings_condition",
                "V16b": "buildings_condition_spec",
                "V17": "urbanization_signs",
                "V18": "observation",
                "V19": "domiciles_estimates"

            },
        )

def get_nui_cat_variables():
    
    return [
        "city",
        "municipality",
        "municipaliyy_group",
        ""
    ]