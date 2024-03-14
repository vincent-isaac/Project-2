import pandas as pd


def _format_test_results(
    result: pd.DataFrame, test: str, test_name: str
) -> pd.DataFrame:
     
    result.rename(columns={"index": "Property"}, inplace=True)
    result["Test"] = test
    result["Test Name"] = test_name
    if "Setting" not in result.columns:
        result["Setting"] = ""
    result = result[["Test", "Test Name", "Data", "Property", "Setting", "Value"]]
    result.reset_index(inplace=True, drop=True)
    return result
