import pandas as pd

def map_features(csv_path):
    df = pd.read_csv(csv_path)

    selected = df[
        [
            "SourceIP",
            "DestinationIP",
            "Protocol",
            "TotalBytes",
            "Duration"
        ]
    ]

    return selected
