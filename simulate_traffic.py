import requests
import pandas as pd

SERVICE_URL = "http://127.0.0.1:9000/"


if __name__ == "__main__":
    test_items = pd.read_csv("data/test.csv").drop(columns=['id']).to_dict('records')[:250]
    for row in test_items:
        res = requests.post(SERVICE_URL, data = row)
    print("[+] Multiple test cases are sent")

