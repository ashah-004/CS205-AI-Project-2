import math
import time
import platform
import subprocess
import matplotlib.pyplot as plt
import numpy as np

class Instance:
    def __init__(self, features, class_label):
        self.features = np.asarray(features, dtype=np.float32) 
        self.class_label = class_label
        
    def __repr__(self):
        return f"Instance(features={self.features}, label={self.class_label})"

def load_dataset(file_path):
    dataset = []
    class_label_map = {} 
    next_int_label = 0   

    try:
        with open(file_path, 'r') as f:
            header_line_skipped = False
            for line in f:
                line = line.strip()
                if not line or line.startswith(';;;'):
                    continue

                if not header_line_skipped and "REGION-CENTROID-COL" in line.upper():
                    header_line_skipped = True
                    continue

                parts = line.split(',')
                if not parts:
                    continue

                try:
                    string_class_label = parts[0].strip().upper()
                    if string_class_label not in class_label_map:
                        class_label_map[string_class_label] = next_int_label
                        next_int_label += 1
                    class_label = class_label_map[string_class_label] 

                    features = [float(p) for p in parts[1:]] 
                    
                    dataset.append(Instance(features, class_label))
                except ValueError as ve:
                    print(f"Warning: Skipping malformed line '{line}'. Error converting feature to float: {ve}")
                    continue
                except IndexError:
                    print(f"Warning: Skipping malformed line '{line}'. Not enough parts after splitting.")
                    continue
        
        if not dataset:
            print(f"Error: Dataset file '{file_path}' is empty or contains no valid data after parsing.")
            return []
            
        print(f"Loaded {len(dataset)} instances.")
        print(f"Categorical labels mapped to integers: {class_label_map}")
        return dataset
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{file_path}'. Please check the path and try again.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading '{file_path}': {e}")
        return []


def euclidean_distance(instance1_features, instance2_features):

    if len(instance1_features) != len(instance2_features):
        raise ValueError("Feature vectors must have the same number of dimensions for Euclidean distance calculation.")

    sum_sq_diff = 0

    for i in range(len(instance1_features)):
        difference = instance1_features[i] - instance2_features[i]
        sum_sq_diff += (difference ** 2)

    return math.sqrt(sum_sq_diff)

def calculate_majority_class_accuracy(dataset):

    if not dataset:
        return 0.0

    class_counts = {}
    for instance in dataset:
        class_counts[instance.class_label] = class_counts.get(instance.class_label, 0) + 1
    
    if not class_counts:
        return 0.0

    majority_class_count = max(class_counts.values())
    return (majority_class_count / len(dataset)) * 100

def nearest_neighbor_accuracy(dataset, current_feature_indices, majority_class_accuracy=0.0):

    if not dataset:
        return 0.0 

    if not current_feature_indices:
        return majority_class_accuracy

    num_correct_predictions = 0 

    feature_indices_list = sorted(list(current_feature_indices))

    for i in range(len(dataset)):
        test_instance = dataset[i]
        test_features_subset = test_instance.features[feature_indices_list]

        min_distance = float('inf') 
        nearest_neighbor_class = -1 

        for j in range(len(dataset)):
            if i == j: 
                continue 

            training_instance = dataset[j]
            training_features_subset = training_instance.features[feature_indices_list]

            distance = euclidean_distance(test_features_subset, training_features_subset)

            if distance < min_distance:
                min_distance = distance
                nearest_neighbor_class = training_instance.class_label

        if nearest_neighbor_class == test_instance.class_label:
            num_correct_predictions += 1

    accuracy = (num_correct_predictions / len(dataset)) * 100
    return accuracy



def forward_selection(dataset, majority_class_accuracy):
    num_total_features = len(dataset[0].features)

    current_features = set() 
    best_overall_features = set() 
    best_overall_accuracy = majority_class_accuracy 

    plot_data = []
    plot_data.append((frozenset(), majority_class_accuracy))

    print(f"Running nearest neighbor with no features (default rate), using \"leaving-one-out\" evaluation, I get an accuracy of {majority_class_accuracy:.2f}%")
    print("Beginning search.\n")

    for i in range(num_total_features):
        print(f"On level {i + 1} of the search tree, considering adding a feature to {set(sorted([f + 1 for f in current_features]))}.")

        feature_to_add_this_level = -1 
        best_accuracy_this_level = -1.0 

        for candidate_feature_idx in range(num_total_features):
            if candidate_feature_idx not in current_features:
                temp_features = current_features.union({candidate_feature_idx})

                accuracy = nearest_neighbor_accuracy(dataset, temp_features, majority_class_accuracy)

                print_temp_features = sorted([f + 1 for f in temp_features])
                print(f"          Using feature(s) {{{', '.join(map(str, print_temp_features))}}} accuracy is {accuracy:.2f}%")

                if accuracy > best_accuracy_this_level:
                    best_accuracy_this_level = accuracy
                    feature_to_add_this_level = candidate_feature_idx

        if feature_to_add_this_level == -1:
            print("Error: No feature could be added (this should not happen in a complete forward selection). Exiting.")
            break

        current_features.add(feature_to_add_this_level)
        print_current_features = sorted([f + 1 for f in current_features])

        print(f"Feature set {set(print_current_features)} was best, accuracy {best_accuracy_this_level:.2f}%\n")

        plot_data.append((frozenset(current_features), best_accuracy_this_level))

        if best_accuracy_this_level > best_overall_accuracy:
            best_overall_accuracy = best_accuracy_this_level
            best_overall_features = current_features.copy() 
        else:
            print(f"(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
            print(f"(Current best feature subset: {set(sorted([f + 1 for f in best_overall_features]))}, accuracy: {best_overall_accuracy:.2f}%)\n")

    print(f"Finished search!! The best feature subset is {set(sorted([f + 1 for f in best_overall_features]))}, which has an accuracy of {best_overall_accuracy:.2f}%")
    return best_overall_features, best_overall_accuracy, plot_data


def backward_elimination(dataset, majority_class_accuracy):
    num_total_features = len(dataset[0].features)

    current_features = set(range(num_total_features))
    initial_all_features_accuracy = nearest_neighbor_accuracy(dataset, current_features, majority_class_accuracy)

    best_overall_features = current_features.copy()
    best_overall_accuracy = initial_all_features_accuracy

    plot_data = []
    plot_data.append((frozenset(current_features), initial_all_features_accuracy))

    print(f"Running nearest neighbor with all {num_total_features} features, using \"leaving-one-out\" evaluation, I get an accuracy of {initial_all_features_accuracy:.2f}%")

    print("\nBeginning search.\n")

    for i in range(num_total_features):
        if not current_features: 
            break 

        print(f"On level {i + 1} of the search tree, considering removing feature from {set(sorted([f + 1 for f in current_features]))}.")

        feature_to_remove_this_level = -1 
        best_accuracy_this_level = -1.0 

        if len(current_features) == 1:
            accuracy = majority_class_accuracy 
            feature_to_remove_this_level = list(current_features)[0] 
            
            print(f"          Using feature(s) {{}} accuracy is {accuracy:.2f}%")
            best_accuracy_this_level = accuracy 
            
        else: 
            for candidate_feature_idx in current_features:
                temp_features = current_features.difference({candidate_feature_idx})

                accuracy = nearest_neighbor_accuracy(dataset, temp_features, majority_class_accuracy)

                print_temp_features = sorted([f + 1 for f in temp_features])
                print(f"          Using feature(s) {{{', '.join(map(str, print_temp_features))}}} accuracy is {accuracy:.2f}%")
                if accuracy > best_accuracy_this_level:
                    best_accuracy_this_level = accuracy
                    feature_to_remove_this_level = candidate_feature_idx

        if feature_to_remove_this_level == -1:
            print("Error: No feature could be identified for removal. Exiting backward elimination prematurely.")
            break

        current_features.remove(feature_to_remove_this_level)
        print_current_features = sorted([f + 1 for f in current_features])
        
        if not print_current_features: 
            print(f"Feature set {{}} was best, accuracy {best_accuracy_this_level:.2f}%\n")
        else:
            print(f"Feature set {set(print_current_features)} was best, accuracy {best_accuracy_this_level:.2f}%\n")
        
        plot_data.append((frozenset(current_features), best_accuracy_this_level))

        if best_accuracy_this_level > best_overall_accuracy:
            best_overall_accuracy = best_accuracy_this_level
            best_overall_features = current_features.copy()
        else:
            print(f"(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
            print(f"(Current best feature subset: {set(sorted([f + 1 for f in best_overall_features]))}, accuracy: {best_overall_accuracy:.2f}%)\n")    
            print(f"Finished search!! The best feature subset is {set(sorted([f + 1 for f in best_overall_features]))}, which has an accuracy of {best_overall_accuracy:.2f}%")
    return best_overall_features, best_overall_accuracy, plot_data



def plot_accuracy_vs_features(plot_data, algorithm_name, dataset_name):

    if not plot_data:
        print("No data to plot.")
        return

    feature_sets_raw = [item[0] for item in plot_data]
    accuracy_list = [item[1] for item in plot_data]

    x_labels = []
    for fs in feature_sets_raw:
        if not fs: 
            x_labels.append("{}")
        else:
            sorted_features_display = sorted([f + 1 for f in fs])
            x_labels.append(f"{{{', '.join(map(str, sorted_features_display))}}}")

    x_positions = list(range(len(plot_data))) 

    fig, ax = plt.subplots(figsize=(15, 8))
    
    bars = ax.bar(x_positions, accuracy_list, color='skyblue', width=0.7)
    
    ax.set_title(f'Accuracy vs. Feature Set\n({algorithm_name} on {dataset_name})', fontsize=16)
    
    if algorithm_name == "Forward Selection":
        ax.set_xlabel('Features Added', fontsize=14)
    elif algorithm_name == "Backward Elimination":
        ax.set_xlabel('Features Remaining', fontsize=14)
    else:
        ax.set_xlabel('Feature Set', fontsize=14)

    ax.set_ylabel('Accuracy (%)', fontsize=14)
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=10)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")

    ax.tick_params(axis='y', labelsize=12)

    max_accuracy_val = max(accuracy_list) if accuracy_list else 100
    y_upper_limit = max(105.0, max_accuracy_val * 1.05)
    ax.set_ylim(0, y_upper_limit)

    for i, bar in enumerate(bars):
        yval = bar.get_height()
        label = f'{yval:.2f}%'
        
        if (yval + 1.5) > (ax.get_ylim()[1] - 0.5): 
            ax.text(bar.get_x() + bar.get_width()/2, yval - 2.5, label,
                    ha='center', va='top', fontsize=9, color='white',
                    weight='bold', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))
        else:
            ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, label,
                    ha='center', va='bottom', fontsize=10, color='black')

    if accuracy_list:
        max_accuracy = max(accuracy_list)
        max_accuracy_idx = -1
        for i, acc in enumerate(accuracy_list):
            if acc == max_accuracy:
                max_accuracy_idx = i
                break
        
        if max_accuracy_idx != -1:
            bars[max_accuracy_idx].set_color('red')

    plt.tight_layout()
    plt.show()


def print_machine_info():
    print("\n--- Machine Information ---")

    print("CPU:")
    try:
        cpu_model = ""
        try:
            cpu_model_output = subprocess.check_output("lscpu | grep 'Model name:'", shell=True).decode().strip()
            cpu_model = cpu_model_output.split(':')[-1].strip()
        except subprocess.CalledProcessError:
            cpu_model = platform.processor()

        print(f"  Model Name: {cpu_model}")
        lscpu_output = subprocess.check_output("lscpu", shell=True).decode()
        lscpu_lines = lscpu_output.splitlines()

        cpu_details = {}
        for line in lscpu_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                cpu_details[key.strip()] = value.strip()

        print(f"  Architecture: {cpu_details.get('Architecture', 'N/A')}")
        print(f"  CPU(s) (Logical): {cpu_details.get('CPU(s)', 'N/A')}")
        print(f"  On-line CPU(s) list: {cpu_details.get('On-line CPU(s) list', 'N/A')}")
        print(f"  Core(s) per socket: {cpu_details.get('Core(s) per socket', 'N/A')}")
        print(f"  Socket(s): {cpu_details.get('Socket(s)', 'N/A')}")
        print(f"  Thread(s) per core: {cpu_details.get('Thread(s) per core', 'N/A')}")
        print(f"  L3 Cache: {cpu_details.get('L3 cache', 'N/A')}") 

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  CPU Info: Details via lscpu not fully available. Error: {e}")
        print(f"  Fallback CPU: {platform.processor()}")

    print("Memory:")
    try:
        mem_info = subprocess.check_output("cat /proc/meminfo | grep 'MemTotal'", shell=True).decode().strip()
        mem_total_kb = int(mem_info.split(':')[1].strip().split(' ')[0])
        mem_total_gb = mem_total_kb / (1024**2)
        print(f"  Total: {mem_total_gb:.1f} GB")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  Total: N/A (Details via /proc/meminfo not available). Error: {e}")

    print("OS:")
    print(f"  System: {platform.system()}")
    print(f"  Release: {platform.release()}")
    print(f"  Version: {platform.version()}")

    print("---------------------------\n")


import numpy as np 

def normalize_features(dataset):
    if not dataset or not dataset[0].features.size > 0:
        print("Dataset is empty or has no features for normalization. Skipping normalization.")
        return dataset

    all_features_matrix = np.array([instance.features for instance in dataset], dtype=np.float32)

    means = np.mean(all_features_matrix, axis=0)
    stds = np.std(all_features_matrix, axis=0)

    stds[stds == 0] = 1.0 

    normalized_features_matrix = (all_features_matrix - means) / stds

    for i, instance in enumerate(dataset):
        instance.features = normalized_features_matrix[i] 

    print("Features normalized (Z-score) successfully.")
    return dataset

algorithm_name_global = ""
plot_data_results_global = []
initial_all_features_accuracy_global = 0.0 
majority_class_accuracy_global = 0.0 
if __name__ == "__main__":
    print("Welcome to Akshat Shah's Feature Selection Algorithm.\n")

    my_dataset = []
    dataset_file = ""
    while not my_dataset: 
        dataset_file = input("Type in the name of the file to test : ")
        my_dataset = load_dataset(dataset_file)
        if not my_dataset:
            print("Please try again with a valid file name.")

    if my_dataset:
        my_dataset = normalize_features(my_dataset)

    num_instances = len(my_dataset)
    num_features = len(my_dataset[0].features) if num_instances > 0 else 0

    if num_features == 0:
        print("Error: The loaded dataset contains no features. Cannot proceed with feature selection.")
        exit()
    if num_instances < 2:
        print("Error: Dataset too small for Leaving-One-Out Cross-Validation (needs at least 2 instances).")
        exit()

    print(f"This dataset has {num_features} features (not including the class attribute), with {num_instances} instances.\n")

    majority_class_accuracy_global = calculate_majority_class_accuracy(my_dataset)
    print(f"The majority class baseline (0 features) accuracy is {majority_class_accuracy_global:.2f}%\n")

    all_feature_indices = set(range(num_features))
    initial_all_features_accuracy_global = nearest_neighbor_accuracy(my_dataset, all_feature_indices, majority_class_accuracy_global)
    print(f"Running nearest neighbor with all {num_features} features, using \"leaving-one-out\" evaluation, I get an accuracy of {initial_all_features_accuracy_global:.2f}%\n")

    choice = ''
    while choice not in ['1', '2']:
        print("\nType the number of the algorithm you want to run.")
        print("1) Forward Selection")
        print("2) Backward Elimination")
        choice = input("Enter choice (1 or 2): ")
        if choice not in ['1', '2']:
            print("Invalid choice. Please enter 1 or 2.")

    start_time = time.time()

    if choice == '1':
        algorithm_name_global = "Forward Selection"
        best_features, best_accuracy, plot_data_results_global = forward_selection(my_dataset, majority_class_accuracy_global)
    elif choice == '2':
        algorithm_name_global = "Backward Elimination"
        best_features, best_accuracy, plot_data_results_global = backward_elimination(my_dataset, majority_class_accuracy_global)
    else:
        print("Invalid choice. This branch should not be reached due to the while loop.")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nAlgorithm finished in {elapsed_time:.2f} seconds.")


