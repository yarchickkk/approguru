import requests
import json
from dataclasses import dataclass


@dataclass
class Parameters():
    pool_address: str = ""
    timeframe: str = "minute"
    aggregate: int = 5
    limit: int = 450
    before_timestamp: int = 0


def get_ohlcvs_coingecko(p: Parameters, title: str) -> None:
    url = f"https://api.geckoterminal.com/api/v2/networks/eth/pools/{p.pool_address}/ohlcv/{p.timeframe}?aggregate={p.aggregate}&before_timestamp={p.before_timestamp}&limit={p.limit}&currency=usd&token=base"
    headers = {"accept": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        if int(response.status_code) == 200:
            data = response.json()
            title = f"data/{title}_data.json"
            with open(title, "w") as json_file:
                json.dump(data, json_file, indent=4)
    except:
        return None
    

if __name__ == "__main__":
    
    parameters = Parameters(pool_address="0x7b73644935b8e68019ac6356c40661e1bc315860",
                            before_timestamp=1731179760)
    get_ohlcvs_coingecko(parameters, title='MAGA')
 