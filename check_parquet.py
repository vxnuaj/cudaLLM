#!/usr/bin/env python3
"""
Quick script to check the parquet file contents and size.
"""

import pandas as pd
import os

def check_parquet_file():
    parquet_path = "prime_cuda_llm/rl_cuda_llm_0424.parquet"
    
    if not os.path.exists(parquet_path):
        print(f"File not found: {parquet_path}")
        return
    
    df = pd.read_parquet(parquet_path)
    print(f"Samples: {len(df)}")

if __name__ == "__main__":
    check_parquet_file()
