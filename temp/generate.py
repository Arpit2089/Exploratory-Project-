import pickle

# Hardcoded data extracted directly from the LEMONADE NAS log
history_data = {
    0: [
        {'params': 114371, 'flops': 17028350, 'val_error': 0.3030},
        {'params': 105802, 'flops': 31025664, 'val_error': 0.3212},
        {'params': 89706,  'flops': 15241856, 'val_error': 0.3450}
    ],
    1: [
        {'params': 106106, 'flops': 31353344, 'val_error': 0.2686},
        {'params': 105802, 'flops': 35381248, 'val_error': 0.2800},
        {'params': 114371, 'flops': 17028350, 'val_error': 0.3030},
        {'params': 105802, 'flops': 31025664, 'val_error': 0.3212},
        {'params': 89706,  'flops': 15241856, 'val_error': 0.3450}
    ],
    2: [
        {'params': 89674,  'flops': 15176320, 'val_error': 0.2374},
        {'params': 82179,  'flops': 14968062, 'val_error': 0.2464},
        {'params': 106106, 'flops': 31353344, 'val_error': 0.2686},
        {'params': 105802, 'flops': 35381248, 'val_error': 0.2800}
    ],
    3: [
        {'params': 108118, 'flops': 37640192, 'val_error': 0.2304},
        {'params': 89674,  'flops': 15176320, 'val_error': 0.2374},
        {'params': 73610,  'flops': 27140096, 'val_error': 0.2448},
        {'params': 82179,  'flops': 14968062, 'val_error': 0.2464}
    ],
    4: [
        {'params': 108422, 'flops': 37967872, 'val_error': 0.2294},
        {'params': 108118, 'flops': 37640192, 'val_error': 0.2304},
        {'params': 89674,  'flops': 15176320, 'val_error': 0.2374},
        {'params': 73610,  'flops': 27140096, 'val_error': 0.2448},
        {'params': 82179,  'flops': 14968062, 'val_error': 0.2464}
    ],
    5: [
        {'params': 73578,  'flops': 27074560, 'val_error': 0.2224},
        {'params': 73546,  'flops': 27107328, 'val_error': 0.2246},
        {'params': 72446,  'flops': 25944064, 'val_error': 0.2364},
        {'params': 89674,  'flops': 15176320, 'val_error': 0.2374},
        {'params': 82179,  'flops': 14968572, 'val_error': 0.2406},
        {'params': 82179,  'flops': 14968062, 'val_error': 0.2464}
    ],
    6: [
        {'params': 73578,  'flops': 27074560, 'val_error': 0.2136},
        {'params': 73578,  'flops': 22718976, 'val_error': 0.2138},
        {'params': 73514,  'flops': 27041792, 'val_error': 0.2156},
        {'params': 73189,  'flops': 26741760, 'val_error': 0.2242},
        {'params': 76110,  'flops': 14392060, 'val_error': 0.2332},
        {'params': 65674,  'flops': 25051136, 'val_error': 0.2338}
    ],
    7: [
        {'params': 73578,  'flops': 27074560, 'val_error': 0.2136},
        {'params': 73578,  'flops': 22718976, 'val_error': 0.2138},
        {'params': 73514,  'flops': 27041792, 'val_error': 0.2156},
        {'params': 76110,  'flops': 14392060, 'val_error': 0.2160},
        {'params': 74206,  'flops': 12442364, 'val_error': 0.2240},
        {'params': 73189,  'flops': 26741760, 'val_error': 0.2242},
        {'params': 65674,  'flops': 25051136, 'val_error': 0.2338}
    ],
    8: [
        {'params': 76110,  'flops': 14781950, 'val_error': 0.1888},
        {'params': 77230,  'flops': 14686972, 'val_error': 0.2064},
        {'params': 73578,  'flops': 27074560, 'val_error': 0.2136},
        {'params': 73578,  'flops': 22718976, 'val_error': 0.2138},
        {'params': 78426,  'flops': 14557004, 'val_error': 0.2146},
        {'params': 74206,  'flops': 12442364, 'val_error': 0.2152},
        {'params': 73514,  'flops': 27041792, 'val_error': 0.2156},
        {'params': 74142,  'flops': 12409596, 'val_error': 0.2186}
    ],
    9: [
        {'params': 76110,  'flops': 14781950, 'val_error': 0.1888},
        {'params': 74206,  'flops': 12442364, 'val_error': 0.2008},
        {'params': 73450,  'flops': 27009024, 'val_error': 0.2050},
        {'params': 73578,  'flops': 22718976, 'val_error': 0.2138},
        {'params': 74142,  'flops': 12409596, 'val_error': 0.2186},
        {'params': 73189,  'flops': 26741760, 'val_error': 0.2242},
        {'params': 41322,  'flops': 18800640, 'val_error': 0.2320},
        {'params': 70544,  'flops': 11490300, 'val_error': 0.2402}
    ],
    10: [
        {'params': 76110,  'flops': 14781950, 'val_error': 0.1888},
        {'params': 73157,  'flops': 26676224, 'val_error': 0.1966},
        {'params': 74206,  'flops': 12442364, 'val_error': 0.2008},
        {'params': 41194,  'flops': 18735104, 'val_error': 0.2108},
        {'params': 74142,  'flops': 12409596, 'val_error': 0.2186},
        {'params': 40158,  'flops': 17604608, 'val_error': 0.2204},
        {'params': 33418,  'flops': 16777216, 'val_error': 0.2348},
        {'params': 45008,  'flops': 9855996,  'val_error': 0.2348}
    ],
    11: [
        {'params': 76110,  'flops': 14781950, 'val_error': 0.1768},
        {'params': 74206,  'flops': 12832254, 'val_error': 0.1780},
        {'params': 73157,  'flops': 26676224, 'val_error': 0.1870},
        {'params': 74206,  'flops': 12442364, 'val_error': 0.2008},
        {'params': 65253,  'flops': 24652800, 'val_error': 0.2096},
        {'params': 41194,  'flops': 18735104, 'val_error': 0.2108},
        {'params': 74142,  'flops': 12409596, 'val_error': 0.2186},
        {'params': 40158,  'flops': 17604608, 'val_error': 0.2204}
    ],
    12: [
        {'params': 76110,  'flops': 14781950, 'val_error': 0.1768},
        {'params': 74206,  'flops': 12832254, 'val_error': 0.1780},
        {'params': 73157,  'flops': 26676224, 'val_error': 0.1870},
        {'params': 74206,  'flops': 12442364, 'val_error': 0.2008},
        {'params': 65630,  'flops': 24017408, 'val_error': 0.2056},
        {'params': 65253,  'flops': 24652800, 'val_error': 0.2096},
        {'params': 41194,  'flops': 18735104, 'val_error': 0.2108},
        {'params': 74142,  'flops': 12409596, 'val_error': 0.2186}
    ],
    13: [
        {'params': 76110,  'flops': 14781950, 'val_error': 0.1768},
        {'params': 74206,  'flops': 12832254, 'val_error': 0.1780},
        {'params': 73125,  'flops': 26610688, 'val_error': 0.1870},
        {'params': 65630,  'flops': 24017408, 'val_error': 0.1962},
        {'params': 74206,  'flops': 12442364, 'val_error': 0.2008},
        {'params': 65253,  'flops': 24652800, 'val_error': 0.2096},
        {'params': 40158,  'flops': 21637632, 'val_error': 0.2104},
        {'params': 41194,  'flops': 18735104, 'val_error': 0.2108}
    ],
    14: [
        {'params': 76110,  'flops': 14781950, 'val_error': 0.1768},
        {'params': 74206,  'flops': 12832254, 'val_error': 0.1780},
        {'params': 73125,  'flops': 26610688, 'val_error': 0.1870},
        {'params': 65630,  'flops': 24017408, 'val_error': 0.1962},
        {'params': 74206,  'flops': 12442364, 'val_error': 0.2008},
        {'params': 65253,  'flops': 24652800, 'val_error': 0.2018},
        {'params': 61577,  'flops': 22978048, 'val_error': 0.2100},
        {'params': 40158,  'flops': 21637632, 'val_error': 0.2104}
    ]
}

output_filename = 'history2.pkl'

# Save the dictionary to a .pkl file
with open(output_filename, 'wb') as f:
    pickle.dump(history_data, f)

print(f"Data successfully hardcoded and saved to {output_filename}")