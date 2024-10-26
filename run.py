if __name__ == "__main__":

    import json
    import time
    import approguru as guru
    
    with open("data/token1_data.json", "r") as file:
        data1 = json.load(file)

    with open("data/token2_data.json", "r") as file:
        data2 = json.load(file)

    with open("data/token3_data.json", "r") as file:
        data3 = json.load(file)

    with open("data/token4_data.json", "r") as file:
        data4 = json.load(file)

    with open("data/token5_data.json", "r") as file:
        data5 = json.load(file)
    
    with open("data/trump_data.json", "r") as file:
        trump_data = json.load(file)

    with open("data/pepe_data.json", "r") as file:
        pepe_data = json.load(file)

    with open("data/spx_data.json", "r") as file:
        spx_data = json.load(file)

    with open("data/mstr_data.json", "r") as file:
        mstr_data = json.load(file)

    with open("data/MARS_data.json", "r") as file:
        mars_data = json.load(file)

    with open("data/NEIRO_data.json", "r") as file:
        neiro_data = json.load(file)

    with open("data/HANA_data.json", "r") as file:
        hana_data = json.load(file)

    with open("data/DOGGO_data.json", "r") as file:
        doggo_data = json.load(file)

    with open("data/JOE_data.json", "r") as file:
        joe_data = json.load(file)

    with open("data/FEFE_data.json", "r") as file:
        fefe_data = json.load(file)

    with open("data/WOJAK_data.json", "r") as file:
        wojak_data = json.load(file)

    with open("data/some_data.json", "r") as file:
        some_data = json.load(file)


    finder = guru.MaxFallFinder()
    bruh = [data1, data2, data3, data4, data5, trump_data, pepe_data, spx_data, mstr_data, neiro_data, hana_data, doggo_data, joe_data, fefe_data, wojak_data, some_data]  # 16
    data_list = []
    for i in range(5):
        data_list += bruh  # 80

    # linear processing:
    start = time.perf_counter()
    for data in data_list:
        finder(data)
        print(f"Max fall: {finder.max_fall}, length: {finder.min_val_idx - finder.max_val_idx}")
    end = time.perf_counter()
    print(f"Linear processing: {end - start:.2f} seconds.")


    # parallel processing:
    start = time.perf_counter()
    results = finder.parallel_process(data_list, num_workers=8)
    end = time.perf_counter()

    for i in results:
        print(f"Max fall: {i[0]}, length: {i[1] - i[2] if (i[1], i[2]) else None}")

    end = time.perf_counter()
    print(f"Parallel processing: {end - start:.2f} seconds.")
  