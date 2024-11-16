import json
import requests
from dataclasses import dataclass


@dataclass
class Parameters():
    pool_address: str = ""
    timeframe: str = "hour"
    aggregate: int = 1
    limit: int = 150
    before_timestamp: str = "1731707759" 


def get_ohlcvs_coingecko(p: Parameters, title: str, timestamp: bool = True) -> None:
    if timestamp is True:
        url = f"https://api.geckoterminal.com/api/v2/networks/eth/pools/{p.pool_address}/ohlcv/{p.timeframe}?aggregate={p.aggregate}&before_timestamp={p.before_timestamp}&limit={p.limit}&currency=usd&token=base"
    elif timestamp is False:
        url = f"https://api.geckoterminal.com/api/v2/networks/eth/pools/{p.pool_address}/ohlcv/{p.timeframe}?aggregate={p.aggregate}&limit={p.limit}&currency=usd&token=base"
    
    headers = {"accept": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        if int(response.status_code) == 200:
            data = response.json()
            title = f"data/{title}_data.json"
            with open(title, "w") as json_file:
                json.dump(data, json_file, indent=4)
    except:
        print("Got no data!")
        return None

pools = [
    "0xa43fe16908251ee70ef74718545e4fe6c5ccec9f",
    "0x308c6fbd6a14881af333649f17f2fde9cd75e2a6",
    "0x6a63be37a5501dc0f88ed3fa21d3cf9a64eb618e",
    "0xc2eab7d33d3cb97692ecb231a5d0e4a649cb539d",
    "0x52c77b0cb827afbad022e6d6caf2c44452edbc39",
    "0xc555d55279023e732ccd32d812114caf5838fd46",
    "0x6c2b607bf0a8ede629d58ba5e05d0448ff7f890a",
    "0xa1bf0e900fb272089c9fd299ea14bfccb1d1c2c0",
    "0x5ced44f03ff443bbe14d8ea23bc24425fb89e3ed",
    "0x318ba85ca49a3b12d3cf9c72cc72b29316971802",
    "0xe945683b3462d2603a18bdfbb19261c6a4f03ad1",
    "0xca7c2771d248dcbe09eabe0ce57a62e18da178c0",
    "0x0f23d49bc92ec52ff591d091b3e16c937034496e",
    "0x470dc172d6502ac930b59322ece5345dd456a03d",
    "0x67324985b5014b36b960273353deb3d96f2f18c2",
    "0xddd23787a6b80a794d952f5fb036d0b31a8e6aff",
    "0x69c7bd26512f52bf6f76fab834140d13dda673ca",
    "0x6bcd2862522c0ab45f4f9fe693e36c791ede0a42"
]  

if __name__ == "__main__":

    # Get OHLCV data
    """for idx, pool in enumerate(pools):
        parameters = Parameters(pool_address=pool)
        get_ohlcvs_coingecko(parameters, title=str(idx))"""
    

    pool = "0xf22fdd2be7c6da9788e4941a6ffc78ca99d7b15c"
    parameters = Parameters(pool_address=pool)
    get_ohlcvs_coingecko(parameters, title=str("test"))
