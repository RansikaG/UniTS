import os
import random
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sktime.datasets import load_from_tsfile_to_dataframe
from pathlib import Path

class TSDataSplitter:
    def __init__(self, input_file, output_dir="cstnet", train_ratio=0.8, bucket_size=50):
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.bucket_size = bucket_size

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_file = self.output_dir / "newdata_TRAIN.ts"
        self.test_file = self.output_dir / "newdata_TEST.ts"

        # Read original data format
        self.headers, self.raw_data, self.raw_labels = self._read_original_format()

    def _read_original_format(self):
        """Read the original file format including headers and raw data."""
        headers = []
        data = []
        labels = []

        with open(self.input_file, 'r') as f:
            # Read headers
            for line in f:
                if line.startswith('@'):
                    headers.append(line.strip())
                    if line.startswith('@data'):
                        break

            # Read data lines
            for line in f:
                if line.strip():  # Skip empty lines
                    # Split at the last colon to separate data and label
                    *data_parts, label = line.strip().rsplit(':', 1)
                    data_line = ':'.join(data_parts)  # Rejoin any colons in the data part
                    data.append(data_line)
                    labels.append(label.strip())

        return headers, data, labels

    def analyze_sequence_lengths(self):
        """Calculate and plot sequence length distribution."""
        lengths = [len(seq.split(',')) for seq in self.raw_data]
        max_length = max(lengths)
        buckets = range(0, max_length + self.bucket_size, self.bucket_size)

        length_freq = Counter(
            (length // self.bucket_size) * self.bucket_size for length in lengths
        )
        print(length_freq)
        return length_freq

    def split_data(self, min_length):
        """Split the data while maintaining original format."""
        # Filter sequences by minimum length
        valid_indices = [i for i, seq in enumerate(self.raw_data)
                         if len(seq.split(',')) >= min_length]

        if not valid_indices:
            raise ValueError(f"No sequences found with length >= {min_length}")

        # Create arrays of valid data
        valid_data = [self.raw_data[i] for i in valid_indices]
        valid_labels = [self.raw_labels[i] for i in valid_indices]

        # Split indices for each class
        train_indices = []
        test_indices = []

        for label in set(valid_labels):
            class_indices = [i for i, l in enumerate(valid_labels) if l == label]
            np.random.shuffle(class_indices)

            split_idx = int(len(class_indices) * self.train_ratio)
            train_indices.extend(class_indices[:split_idx])
            test_indices.extend(class_indices[split_idx:])

        # Create final datasets
        train_data = [valid_data[i] for i in train_indices]
        train_labels = [valid_labels[i] for i in train_indices]
        test_data = [valid_data[i] for i in test_indices]
        test_labels = [valid_labels[i] for i in test_indices]

        return train_data, train_labels, test_data, test_labels

    def save_to_ts_file(self, data_sequences, labels, output_file):
        """Save the split data to a .ts file in original format."""
        try:
            with open(output_file, 'w') as f:
                # Write original headers
                for header in self.headers:
                    if header.startswith('@classLabel'):
                        # Update class label line with current unique labels
                        f.write('@classLabel true ' + ' '.join(set(labels)) + '\n')
                    else:
                        f.write(header + '\n')

                # Write data lines in original format
                for seq, label in zip(data_sequences, labels):
                    f.write(f'{seq}:{label}\n')

        except Exception as e:
            raise RuntimeError(f"Error saving data to {output_file}: {str(e)}")

    def process(self, min_length=None):
        """Process the data: analyze, split, and save."""
        # Analyze and show distribution
        length_freq = self.analyze_sequence_lengths()

        # Get minimum length if not provided
        if min_length is None:
            min_length = int(input("Please enter the minimum sequence length: "))

        # Split the data
        train_data, train_labels, test_data, test_labels = self.split_data(min_length)

        # Save split data
        self.save_to_ts_file(train_data, train_labels, self.train_file)
        self.save_to_ts_file(test_data, test_labels, self.test_file)

        print(f"Training data saved to: {self.train_file}")
        print(f"Testing data saved to: {self.test_file}")

        # Return split information
        return {
            'train_size': len(train_data),
            'test_size': len(test_data),
            'train_class_dist': Counter(train_labels),
            'test_class_dist': Counter(test_labels)
        }

def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create splitter instance
    splitter = TSDataSplitter(
        input_file="cstnet.ts",
        output_dir="cstnet",
        train_ratio=0.8,
        bucket_size=50
    )

    # Process the data
    try:
        results = splitter.process()
        print("\nSplit Results:")
        print(f"Training set size: {results['train_size']}")
        print(f"Testing set size: {results['test_size']}")
        print("\nClass distribution in training set:", dict(results['train_class_dist']))
        print("Class distribution in testing set:", dict(results['test_class_dist']))
    except Exception as e:
        print(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    main()