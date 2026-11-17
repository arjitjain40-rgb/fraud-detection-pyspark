import os
import time
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
STREAM_INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "stream_input")


def simulate_stream(batch_size=5, interval_seconds=4, num_batches=10):
    os.makedirs(STREAM_INPUT_DIR, exist_ok=True)

    print(f"Loading sample data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    fraud_df = df[df["Class"] == 1]
    normal_df = df[df["Class"] == 0]

    print(f"Total transactions available: {len(df)}")
    print(f"Starting real-time simulation: {num_batches} batches of {batch_size} transactions...")
    print(f"Dropping files into: {STREAM_INPUT_DIR}\n")

    for i in range(1, num_batches + 1):
        # Inject 1 fraud transaction every other batch to test detection
        if i % 2 == 0 and len(fraud_df) > 0:
            sample_fraud = fraud_df.sample(n=1)
            sample_normal = normal_df.sample(n=batch_size - 1)
            batch = pd.concat([sample_fraud, sample_normal]).sample(frac=1)
            status = "(contains simulated FRAUD)"
        else:
            batch = normal_df.sample(n=batch_size)
            status = "(all legitimate)"

        file_name = f"tx_stream_batch_{i}_{int(time.time())}.csv"
        file_path = os.path.join(STREAM_INPUT_DIR, file_name)

        batch.to_csv(file_path, index=False)
        print(f"[Batch {i}/{num_batches}] Generated {batch_size} transactions {status} -> {file_name}")

        time.sleep(interval_seconds)

    print("\nSimulation complete! All batches sent.")


if __name__ == "__main__":
    simulate_stream()
