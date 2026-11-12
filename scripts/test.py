from sklearn.datasets import fetch_openml
raw = fetch_openml(data_id=1597, as_frame=True, parser="auto")
print("ignore:", raw.details.get("ignore_attribute"))
print("row_id:", raw.details.get("row_id_attribute"))