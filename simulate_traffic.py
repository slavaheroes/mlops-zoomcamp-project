import requests
import pandas as pd
from utils import config


if __name__ == "__main__":
    test_items = pd.read_csv("data/test.csv").drop(columns=['id']).to_dict('records')[:250]
    for row in test_items:
        res = requests.post(config["SERVICE_URL"], data = row)
    print("[+] Multiple test cases are sent")

