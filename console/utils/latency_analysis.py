#!/usr/bin/env python
from typing import List

import argparse
import time
import urllib
import requests

import numpy as np
import pandas as pd
from tqdm import tqdm

# Measure the latency of model predictions of the running server
def get_args() -> argparse.ArgumentParser:
    '''
    Return CLI configuration for running latency analysis
    '''
    parser = argparse.ArgumentParser(description='Run a latency analysis')
    parser.add_argument(
        '--model_server',
        default="http://localhost:8080"
    )

    parser.add_argument(
        "--data_file",
        default="data.csv",
        help="the file to read data from"
    )

    parser.add_argument(
        "--warmup",
        default=1,
        help="Number of iterations to warmup the server"
    )

    return parser

def time_utterance(utterance: str, model_server: str) -> float:
    """
    Measure how long it takes for an utterance to
    execute
    """
    time_start: float = time.time()
    payload = {'doc': str(utterance)}
    payload = urllib.parse.urlencode(payload, quote_via=urllib.parse.quote)
    r = requests.get(model_server, params=payload)
    intent_scores = r.text.split("\n")
    time_end: float = time.time()

    return time_end - time_start

def read_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, sep='\t')

def execute_dataset(data: pd.DataFrame, model_server: str) -> List[float]:
    time_taken: List[float] = []
    print("starting executing analysis")
    for row in tqdm(data.iterrows()):
        utterance: str = None
        utterance_time: float = time_utterance(utterance, model_server)
        time_taken.append(utterance_time)
    
    return time_taken

def main() -> None:
    args = get_args().parse_args()
    data: pd.DataFrame = read_csv(args.data_file) 
    
    print("starting warmup iterations")
    for i in range(args.warmup):
        print(f"executing warmup iteration: {i}")
        execute_dataset(data, args.model_server)
    
    print("begin analysis")
    time_taken_list = execute_dataset(data, args.model_server)
    print(f"========================")
    print(f"Total time: {np.sum(time_taken_list):.2f} ms")
    print(f"Mean time: {np.mean(time_taken_list):.2f} ms")
    print(f"P50: {np.percentile(time_taken_list, 50):.2f} ms")
    print(f"P90: {np.percentile(time_taken_list, 90):.2f} ms")
    print(f"P95: {np.percentile(time_taken_list, 95):.2f} ms")
    print(f"P99: {np.percentile(time_taken_list, 99):.2f} ms")


if __name__ == "__main__":
    main()
