import requests
import json
from dataclasses import dataclass


@dataclass
class Parameters():
    pool_address: str = ""
    timeframe: str = "hour"
    aggregate: int = 1
    limit: int = 150


def get_ohlcvs_coingecko(p: Parameters, title: str) -> None:
    url = f"https://api.geckoterminal.com/api/v2/networks/eth/pools/{p.pool_address}/ohlcv/{p.timeframe}?aggregate={p.aggregate}&before_timestamp=1731533108&limit={p.limit}&currency=usd&token=base"
    headers = {"accept": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        print(response.status_code)
        if int(response.status_code) == 200:
            data = response.json()
            title = f"data/{title}_data.json"
            with open(title, "w") as json_file:
                json.dump(data, json_file, indent=4)
    except:
        print("aaa")
        return None
    

if __name__ == "__main__":
    
    parameters = Parameters(pool_address="0x9b9d97a9215882f4a68b0d0d591d5aaeb29e5e47")
    get_ohlcvs_coingecko(parameters, title='HOUR')

# POOL: 0xcbe856765eeec3fdc505ddebf9dc612da995e593
# DEBUG AT TIMESTAMP: 1731438368
# guru glues together some extremums (why?)
 