import json
import requests
from dataclasses import dataclass


@dataclass
class Parameters():
    pool_address: str = ""
    timeframe: str = "minute"
    aggregate: int = 15
    limit: int = 250
    before_timestamp: str = "1731927275" 


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

'''pools = [
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
]  '''

pools = [
    "0x6bcd2862522c0ab45f4f9fe693e36c791ede0a42",
    "0xf22fdd2be7c6da9788e4941a6ffc78ca99d7b15c",
    "0x29c830864930c897efa2b9e9851342187b82010e",
    "0xdeba8fd61c1c87b6321a501ebb19e61e610421bf",
    "0xd9f2a7471d1998c69de5cae6df5d3f070f01df9f",
    "0x8a85f9384c37ecc64d33c37e490d7b4e5951cb59",
    "0xc0bef58a0fec1b694c752e0651b315acb147ee6e",
    "0x03296fdd8bf8e0f9def07fac3466b256fc8720b0",
    "0x877193009a881359e80df3ded7c6e055be9cc144",
    "0xfdb676331a4b37689b2ea12f14ff895b640a4379",
    "0x47082a75bc16313ef92cfaca1feb885659c3c9b5",
    "0x0f388ecdbc128083bfe317b00de2c25fbe9f24d5",
    "0x7c2543938a27554f82be178a2ac02a72c8ad1925",
    "0x55d5c232d921b9eaa6b37b5845e439acd04b4dba",
    "0x750874e6fb8dca30ce41d445e4baf8c76971f912",
    "0x1f4ef1f8441caac34f58fb0cba813dd2b09fec63",
    "0x9b9d97a9215882f4a68b0d0d591d5aaeb29e5e47",
    "0xf94f040ad12abd0b585eae84876429715bfa82d6",
    "0xe7b4e528308c84fd6698906b6224615e9e30d236",
    "0x2325e3f261cadb1c30cebf66c9f95f6fb016c0d4",
    "0xff70de5183aede4be2eff73efcc8ea2a8590229b",
    "0x7eb6d3466600b4857eb60a19e0d2115e65aa815e",
    "0x54f5044efd8538c41ccd4bffb06cf375c9bbb6c4",
    "0x8fbd26a7cb1ab65834c5ea245aa1f1e78d03ed30",
    "0xc98b2d550d8d123f8e6950e0758305e88511b037"
]

if __name__ == "__main__":

    # # Get OHLCV data
    # for idx, pool in enumerate(pools):
    #     parameters = Parameters(pool_address=pool)
    #     get_ohlcvs_coingecko(parameters, title=str(idx))
    

    pool = "0x9ec9620e1fda9c1e57c46782bc3232903cacb59b"
    parameters = Parameters(pool_address=pool)
    get_ohlcvs_coingecko(parameters, title=str("test"), timestamp=False)
