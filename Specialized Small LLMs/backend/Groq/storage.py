import json
import csv
from pathlib import Path

class DataStorage:
    def __init__(self, output_file):
        self.output_file = output_file
        self.data = []
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    def add_entry(self, entry):
        self.data.append(entry)
        if len(self.data) % 100 == 0:
            self._save_batch()

    def _save_batch(self):
        with open(self.output_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"Saved {len(self.data)} entries")

    def final_save(self):
        self._save_batch()
        # Create CSV version
        csv_file = self.output_file.replace('.json', '.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
            writer.writeheader()
            writer.writerows(self.data)