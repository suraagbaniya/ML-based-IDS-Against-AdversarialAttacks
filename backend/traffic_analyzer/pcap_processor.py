import subprocess

def run_networkminer(pcap_path):
    """
    Runs NetworkMiner CLI to extract CSV
    """
    subprocess.run([
        "NetworkMiner.exe",
        "--pcap", pcap_path,
        "--output", "data/processed/"
    ])
