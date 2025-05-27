class Instance:

    def __init__(self, features, class_label):

        self.features = features
        self.class_label = class_label

    def __repr__(self):

        return f"Instance(features={self.features}, label={self.class_label})"
    
def load_dataset(file_path):

    dataset = []

    try:

        with open(file_path, 'r') as f:
            for line in f:

                parts = line.strip().split()

                if not parts:
                    continue

                try:
                    values = [float(p) for p in parts]
                except ValueError:
                    print(f"Warning: Skipping malformed line '{line.strip()}' in '{file_path}'. Non-numeric data found.")
                    continue 

                class_label = int(values[0])

                features = values[1:]

                dataset.append(Instance(features, class_label))

        print(f"Successfully loaded {len(dataset)} instances from '{file_path}'.")
        return dataset

    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{file_path}'. Please check the path.")
        return [] 
    except Exception as e:
        print(f"An unexpected error occurred while loading '{file_path}': {e}")
        return []
