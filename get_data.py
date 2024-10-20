import requests
import json
from pprint import pprint
from dataclasses import dataclass


@dataclass
class Parameters():
    pool_address: str = ""
    timeframe: str = "minute"
    aggregate: int = 5
    before_timestamp: int = 0 #1729350000  # 2024-10-19 18:00:00
    limit: int = 288


def get_ohlcvs_coingecko(p: Parameters, title: str) -> None:
    url = f"https://api.geckoterminal.com/api/v2/networks/eth/pools/{p.pool_address}/ohlcv/{p.timeframe}?aggregate={p.aggregate}&limit={p.limit}&currency=usd&token=base"
    headers = {"accept": "application/json"}

    try:
        response = requests.get(url, headers=headers)
        if int(response.status_code) == 200:
            data = response.json()
            title = f"data/{title}_data.json"
            with open(title, "w") as json_file:
                json.dump(data, json_file, indent=4)
            print(f"{title} loaded, starting at {p.before_timestamp} timestamp")
    except:
        return None


if __name__ == "__main__":
    
    trump_params = Parameters(pool_address="0xe4b8583ccb95b25737c016ac88e539d0605949e8")
    get_ohlcvs_coingecko(trump_params, title='trump')
 
    pepe_params = Parameters(pool_address="0xa43fe16908251ee70ef74718545e4fe6c5ccec9f")
    get_ohlcvs_coingecko(pepe_params, title='pepe')

    spx_params = Parameters(pool_address="0x52c77b0cb827afbad022e6d6caf2c44452edbc39")
    get_ohlcvs_coingecko(spx_params, title='spx')

    mstr_params = Parameters(pool_address="0x318ba85ca49a3b12d3cf9c72cc72b29316971802", before_timestamp="")
    get_ohlcvs_coingecko(mstr_params, title='mstr')
