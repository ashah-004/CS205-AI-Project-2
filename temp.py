import math
import time   
import platform 
import subprocess 
import matplotlib.pyplot as plt

class Instance:
    """
    Represents a single data instance (a row) from the dataset.
    It encapsulates the feature values and its corresponding class label.
    """
    def __init__(self, features, class_label):
        """
        Constructor for the Instance class.

        Args:
            features (list[float]): A list of numerical feature values for this instance.
                                    These are the actual independent variables.
            class_label (int): The classification label for this instance.
                                Based on the problem, this will be 1 or 2.
        """
        self.features = features
        self.class_label = class_label

    def __repr__(self):
        """
        Returns a string representation of the Instance object.
        This is incredibly useful for debugging, as it defines what gets printed
        when you print an Instance object directly.
        """
        return f"Instance(features={self.features}, label={self.class_label})"

def read_instances_from_file(file_path):
    instances = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()  # split by any whitespace
            class_label = int(float(parts[0]))  # convert first value to int label
            features = list(map(float, parts[1:]))  # convert the rest to float features
            instance = Instance(features, class_label)
            instances.append(instance)
    
    return instances

# Example usage
file_path = 'CS205_small_Data__11.txt'  # replace with your actual file name
instances = read_instances_from_file(file_path)

# Print each instance to see features and label
for inst in instances:
    print(inst)


def load_dataset(file_path):
    """
    Loads a dataset from a specified text file into a list of Instance objects.

    Assumptions for the input file format:
    -   Each line represents one data instance.
    -   Values within a line are separated by whitespace (spaces, tabs).
    -   All values are numerical (can be integers, floats, or scientific notation).
    -   The FIRST column is the class label (expected to be 1 or 2).
    -   All SUBSEQUENT columns are feature values.

    Args:
        file_path (str): The absolute or relative path to the dataset text file.

    Returns:
        list[Instance]: A list containing Instance objects, each representing
                        a row from the dataset. Returns an empty list if loading fails.
    """
    dataset = [] # Initialize an empty list to store our Instance objects

    try:
        # Open the file in read mode ('r').
        # The 'with' statement ensures the file is automatically closed
        # even if errors occur, preventing resource leaks.
        with open(file_path, 'r') as f:
            # Iterate over each line in the opened file
            for line in f:
                # Remove leading/trailing whitespace (like newline characters)
                # and then split the string by any whitespace into a list of strings.
                parts = line.strip().split()

                # If the line was empty or only contained whitespace, 'parts' will be empty.
                # Skip such lines.
                if not parts:
                    continue

                # Convert all string parts to floating-point numbers.
                # This handles integers, decimals, and scientific notation (e.g., 1.23e+010).
                try:
                    values = [float(p) for p in parts]
                except ValueError:
                    # If any part cannot be converted to float, it's a malformed line.
                    print(f"Warning: Skipping malformed line '{line.strip()}' in '{file_path}'. Non-numeric data found.")
                    continue # Skip to the next line

                # --- Crucial step based on the provided input format ---
                # The first value in the line is the class label.
                # Convert it to an integer as class labels are typically discrete.
                class_label = int(values[0])

                # All remaining values (from the second element to the end) are features.
                # Use list slicing to get all elements from index 1 onwards.
                features = values[1:]

                # Create a new Instance object and add it to our dataset list.
                dataset.append(Instance(features, class_label))

        print(f"Successfully loaded {len(dataset)} instances from '{file_path}'.\n")
        return dataset

    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{file_path}'. Please check the path.")
        return [] # Return an empty list to indicate failure
    except Exception as e: # Catch any other unexpected errors during file processing
        print(f"An unexpected error occurred while loading '{file_path}': {e}")
        return []


def euclidean_distance(instance1_features, instance2_features):
    """
    Calculates the Euclidean distance between two feature vectors.
    The Euclidean distance is the straight-line distance between two points
    in Euclidean space.

    Formula for n-dimensional space:
    d(A, B) = sqrt( (A1-B1)^2 + (A2-B2)^2 + ... + (An-Bn)^2 )

    Args:
        instance1_features (list[float]): A list of numerical feature values for the first point.
        instance2_features (list[float]): A list of numerical feature values for the second point.

    Returns:
        float: The calculated Euclidean distance.
    Raises:
        ValueError: If the input feature vectors do not have the same number of dimensions.
    """
    # Basic validation: ensure both feature vectors have the same number of dimensions.
    if len(instance1_features) != len(instance2_features):
        raise ValueError("Feature vectors must have the same number of dimensions for Euclidean distance calculation.")

    sum_sq_diff = 0 # Initialize a variable to accumulate the sum of squared differences

    # Iterate through the feature values, calculating the squared difference for each pair.
    for i in range(len(instance1_features)):
        difference = instance1_features[i] - instance2_features[i]
        sum_sq_diff += (difference ** 2) # Add the square of the difference

    # Return the square root of the total sum of squared differences.
    return math.sqrt(sum_sq_diff)

def nearest_neighbor_accuracy(dataset, current_feature_indices):
    """
    Calculates the classification accuracy of a Nearest Neighbor (1-NN) classifier
    using the "Leaving-One-Out" Cross-Validation (LOOCV) method.

    LOOCV works by:
    1.  Taking each instance in the dataset as the "test" instance.
    2.  Using ALL *other* instances as the "training" set.
    3.  Finding the single nearest neighbor to the "test" instance within the "training" set.
    4.  Predicting the class of the "test" instance based on its nearest neighbor's class.
    5.  Comparing the prediction with the actual class.

    This process is repeated for every instance in the dataset.

    Args:
        dataset (list[Instance]): The complete list of Instance objects loaded from the file.
        current_feature_indices (set[int]): A set of 0-based integer indices
                                            representing the features to be used
                                            for distance calculations in this evaluation.
                                            (e.g., {0, 2} means use the 1st and 3rd features
                                            from the `Instance.features` list).

    Returns:
        float: The calculated accuracy as a percentage (0.0 to 100.0).
                Returns 0.0 if the dataset is empty or no features are selected.
    """
    # Handle edge cases: empty dataset or no features selected.
    if not dataset:
        print("Warning: Dataset is empty. Cannot calculate accuracy. Returning 0%.")
        return 0.0
    if not current_feature_indices:
        print("Warning: No features selected for accuracy calculation. Returning 0%.")
        return 0.0

    num_correct_predictions = 0 # Counter for how many instances were correctly classified

    # Outer loop: Iterate through each instance in the dataset.
    # In LOOCV, each instance takes a turn being the "test_instance".
    for i in range(len(dataset)):
        test_instance = dataset[i]

        # Extract only the selected features for the 'test_instance'.
        # This list comprehension builds a new list containing only the feature values
        # whose indices are present in `current_feature_indices`.
        test_features_subset = [test_instance.features[idx] for idx in current_feature_indices]

        min_distance = float('inf') # Initialize with an infinitely large distance
        nearest_neighbor_class = -1 # Placeholder for the class of the nearest neighbor

        # Inner loop: Iterate through all other instances in the dataset to find the nearest neighbor.
        for j in range(len(dataset)):
            # Crucial LOOCV step: Do NOT compare an instance with itself.
            # The 'test_instance' is 'left out' of the 'training set'.
            if i == j:
                continue # Skip the instance that is currently being tested

            training_instance = dataset[j] # This instance is part of the "training set"

            # Extract only the selected features for the 'training_instance'.
            training_features_subset = [training_instance.features[idx] for idx in current_feature_indices]

            # Calculate the Euclidean distance between the test instance's selected features
            # and the current training instance's selected features.
            distance = euclidean_distance(test_features_subset, training_features_subset)

            # Check if this 'training_instance' is closer than any found so far.
            if distance < min_distance:
                min_distance = distance # Update the minimum distance
                nearest_neighbor_class = training_instance.class_label # Store its class label
            # Note on tie-breaking: If two instances are equidistant, this implementation
            # will simply take the first one encountered that matches the `min_distance`.
            # For this project, this is typically acceptable.

        # After checking all other instances, 'nearest_neighbor_class' holds the
        # predicted class for `test_instance`.
        
        # Compare the predicted class with the actual class of the test instance.
        if nearest_neighbor_class == test_instance.class_label:
            num_correct_predictions += 1 # Increment correct predictions if they match

    # Calculate the final accuracy as a percentage.
    accuracy = (num_correct_predictions / len(dataset)) * 100
    return accuracy


def forward_selection(dataset, initial_all_features_accuracy):
    num_total_features = len(dataset[0].features)
    
    current_features = set()
    best_overall_features = set()
    best_overall_accuracy = initial_all_features_accuracy

    plot_data = []
    plot_data.append((frozenset(), initial_all_features_accuracy)) 

    print("Beginning search.\n")

    for i in range(num_total_features):
        print(f"On level {i + 1} of the search tree, considering adding feature to {set(sorted([f + 1 for f in current_features]))}.") 

        feature_to_add_this_level = -1
        best_accuracy_this_level = -1.0
        
        for candidate_feature_idx in range(num_total_features):
            if candidate_feature_idx not in current_features:
                temp_features = current_features.union({candidate_feature_idx})
                
                accuracy = nearest_neighbor_accuracy(dataset, temp_features)
                
                # --- MODIFIED: Print the full temp_features set ---
                print_temp_features = sorted([f + 1 for f in temp_features])
                print(f"        Using feature(s) {{{', '.join(map(str, print_temp_features))}}} accuracy is {accuracy:.2f}%")

                if accuracy > best_accuracy_this_level:
                    best_accuracy_this_level = accuracy
                    feature_to_add_this_level = candidate_feature_idx
        
        if feature_to_add_this_level == -1:
            print("Error: No feature could be added. Exiting forward selection prematurely.")
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


def backward_elimination(dataset):
    num_total_features = len(dataset[0].features)
    
    current_features = set(range(num_total_features))
    
    initial_accuracy = nearest_neighbor_accuracy(dataset, current_features)
    
    best_overall_features = current_features.copy()
    best_overall_accuracy = initial_accuracy 
    
    plot_data = []
    plot_data.append((frozenset(current_features), initial_accuracy)) 

    print(f"Running nearest neighbor with all {num_total_features} features, using \"leaving-one-out\" evaluation, I get an accuracy of {initial_accuracy:.2f}%")
    
    print("\nBeginning search.\n")

    for i in range(num_total_features - 1):
        if len(current_features) <= 1:
            break

        print(f"On level {i + 1} of the search tree, considering removing feature from {set(sorted([f + 1 for f in current_features]))}.")

        feature_to_remove_this_level = -1
        best_accuracy_this_level = -1.0 

        for candidate_feature_idx in current_features:
            temp_features = current_features.difference({candidate_feature_idx})

            if not temp_features:
                accuracy = 0.0
            else:
                accuracy = nearest_neighbor_accuracy(dataset, temp_features)
            
            # --- MODIFIED: Print the full temp_features set ---
            print_temp_features = sorted([f + 1 for f in temp_features])
            print(f"        Using feature(s) {{{', '.join(map(str, print_temp_features))}}} accuracy is {accuracy:.2f}%")

            if accuracy > best_accuracy_this_level:
                best_accuracy_this_level = accuracy
                feature_to_remove_this_level = candidate_feature_idx
        
        if feature_to_remove_this_level == -1:
            print("Error: No feature could be removed. Exiting backward elimination prematurely.")
            break
            
        current_features.remove(feature_to_remove_this_level)
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

import matplotlib.pyplot as plt
import os # Make sure os is imported if you're using os.path.basename later

def plot_accuracy_vs_features(plot_data, algorithm_name, dataset_name):
    """
    Generates a bar graph of accuracy versus feature sets,
    with improved label placement and transparent grid lines.
    Adjusts bar order for backward elimination to show decreasing features.
    """
    if not plot_data:
        print("No data to plot.")
        return

    # --- REMOVED: plot_data.sort(...) ---
    # The plot_data must retain the order from the selection algorithm (forward or backward).
    # Sorting here would destroy the intended order for backward elimination.

    # Prepare data for plotting
    # item[0] is the feature set (frozenset), item[1] is the accuracy
    feature_sets_raw = [item[0] for item in plot_data]
    accuracy_list = [item[1] for item in plot_data]

    # Convert feature sets to display strings for x-axis labels
    x_labels = []
    for fs in feature_sets_raw:
        if not fs: # For the empty set (forward selection start)
            x_labels.append("{}") 
        else:
            # Convert 0-indexed features to 1-indexed for display, then sort
            sorted_features_display = sorted([f + 1 for f in fs])
            x_labels.append(f"{{{', '.join(map(str, sorted_features_display))}}}")

    # --- NEW: Conditional Reversal for Backward Elimination ---
    if algorithm_name == "Backward Elimination":
        # Reverse the order of all lists so bars are plotted from N features down to 1
        feature_sets_raw.reverse()
        accuracy_list.reverse()
        x_labels.reverse()
        # Note: x_positions will be generated based on the reversed lists' new indices
        # So x_positions = list(range(len(plot_data))) remains correct after reversal.
    # --- END NEW ---

    x_positions = list(range(len(plot_data))) # Numerical positions for bars (0, 1, 2, ...)

    # Use ax for better control over plot elements
    fig, ax = plt.subplots(figsize=(15, 8)) # Increased width to accommodate long labels
    
    bars = ax.bar(x_positions, accuracy_list, color='skyblue', width=0.7)
    
    ax.set_title(f'Accuracy vs. Feature Set\n({algorithm_name} on {dataset_name})', fontsize=16)
    
    # Adjust x-axis label based on algorithm
    if algorithm_name == "Forward Selection":
        ax.set_xlabel('Feature Set Added', fontsize=14)
    elif algorithm_name == "Backward Elimination":
        ax.set_xlabel('Feature Set Remaining', fontsize=14)
    else:
        ax.set_xlabel('Feature Set', fontsize=14)

    ax.set_ylabel('Accuracy (%)', fontsize=14)
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Set x-ticks with the string labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=10) # Reduced font size for labels
    
    # Rotate x-axis labels for readability if they are long
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")

    ax.tick_params(axis='y', labelsize=12)

    # --- MODIFIED: More margin above 100% ---
    max_accuracy_val = max(accuracy_list) if accuracy_list else 100
    y_upper_limit = max(105.0, max_accuracy_val * 1.05) 
    ax.set_ylim(0, y_upper_limit)

    # Add accuracy labels on top or inside each bar based on available space
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

    # Highlight the best accuracy bar
    if accuracy_list:
        max_accuracy = max(accuracy_list)
        max_accuracy_idx = -1
        # Find the index of the first occurrence of the max accuracy
        for i, acc in enumerate(accuracy_list):
            if acc == max_accuracy:
                max_accuracy_idx = i
                break
        
        if max_accuracy_idx != -1:
            bars[max_accuracy_idx].set_color('red')

    plt.tight_layout()
    plt.show()

def print_machine_info():
    """
    Prints system information relevant to a Colab-like Linux environment,
    including detailed CPU and memory information.
    """
    print("\n--- Machine Information ---")

    # CPU Info
    print("CPU:")
    try:
        # Get general CPU model name
        cpu_model = ""
        try:
            cpu_model_output = subprocess.check_output("lscpu | grep 'Model name:'", shell=True).decode().strip()
            cpu_model = cpu_model_output.split(':')[-1].strip()
        except subprocess.CalledProcessError:
            cpu_model = platform.processor() # Fallback for model name

        print(f"  Model Name: {cpu_model}")

        # Get detailed CPU counts and architecture info
        lscpu_output = subprocess.check_output("lscpu", shell=True).decode()
        lscpu_lines = lscpu_output.splitlines()

        cpu_details = {}
        for line in lscpu_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                cpu_details[key.strip()] = value.strip()

        # Extract specific CPU details
        print(f"  Architecture: {cpu_details.get('Architecture', 'N/A')}")
        print(f"  CPU(s) (Logical): {cpu_details.get('CPU(s)', 'N/A')}")
        print(f"  On-line CPU(s) list: {cpu_details.get('On-line CPU(s) list', 'N/A')}")
        print(f"  Core(s) per socket: {cpu_details.get('Core(s) per socket', 'N/A')}")
        print(f"  Socket(s): {cpu_details.get('Socket(s)', 'N/A')}")
        print(f"  Thread(s) per core: {cpu_details.get('Thread(s) per core', 'N/A')}")
        print(f"  L3 Cache: {cpu_details.get('L3 cache', 'N/A')}") # Often the most relevant cache size

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  CPU Info: Details via lscpu not fully available. Error: {e}")
        print(f"  Fallback CPU: {platform.processor()}")


    # Memory Info
    print("Memory:")
    try:
        mem_info = subprocess.check_output("cat /proc/meminfo | grep 'MemTotal'", shell=True).decode().strip()
        mem_total_kb = int(mem_info.split(':')[1].strip().split(' ')[0])
        mem_total_gb = mem_total_kb / (1024**2)
        print(f"  Total: {mem_total_gb:.1f} GB")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  Total: N/A (Details via /proc/meminfo not available). Error: {e}")

    # OS Info
    print("OS:")
    print(f"  System: {platform.system()}")
    print(f"  Release: {platform.release()}")
    print(f"  Version: {platform.version()}")

    print("---------------------------\n")


algorithm_name = ""
plot_data_results = []
if __name__ == "__main__":
    # --- 1. Welcome Message ---
    print("Welcome to Akshat Shah's Feature Selection Algorithm.\n")

    # --- 2. Ask for File Name ---
    dataset_file = input("Type in the name of the file to test : ")
    print()
    # For testing, you might still want to uncomment dummy data creation locally
    # if you don't have the actual files immediately available.
    # If running with dummy data, ensure it matches the 12 features, 500 instances
    # to mimic your sample output for testing purposes.
    # Example for dummy data creation (REMOVE FOR FINAL SUBMISSION):
    # import random
    # num_instances_dummy = 500
    # num_features_dummy = 12
    # with open(dataset_file, 'w') as f:
    #     for _ in range(num_instances_dummy):
    #         class_label = random.choice([1, 2])
    #         features = [random.uniform(0, 100) for _ in range(num_features_dummy)]
    #         if class_label == 1:
    #             features[0] += 50
    #             features[5] += 60
    #         else:
    #             features[0] -= 50
    #             features[5] -= 60
    #         f.write(f"{class_label:.1f} " + " ".join([f"{val:.7e}" for val in features]) + "\n")


    # --- 3. Load the dataset ---
    my_dataset = load_dataset(dataset_file)

    while not my_dataset: # Loop until a valid, non-empty dataset is loaded
      dataset_file = input("Type in the name of the file to test : ")
      my_dataset = load_dataset(dataset_file)

    num_instances = len(my_dataset)
    # Handle empty dataset case gracefully for feature count
    num_features = len(my_dataset[0].features) if num_instances > 0 else 0

    # --- 4. Dataset Description ---
    print(f"This dataset has {num_features} features (not including the class attribute), with {num_instances} instances.\n")

    # --- 5. Initial Nearest Neighbor Accuracy (for ALL features) ---
    # This comes AFTER dataset description, BEFORE algorithm choice, per your sample.
    all_feature_indices = set(range(num_features))
    initial_all_features_accuracy = nearest_neighbor_accuracy(my_dataset, all_feature_indices)
    print(f"Running nearest neighbor with all {num_features} features, using \"leaving-one-out\" evaluation, I get an accuracy of {initial_all_features_accuracy:.2f}%\n")


    # --- 6. Algorithm Selection ---
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
        # Forward Selection will print its "Beginning search." internally.
        algorithm_name = "Forward Selection"
        best_features, best_accuracy, plot_data_results = forward_selection(my_dataset, initial_all_features_accuracy)
    elif choice == '2':
        # Backward Elimination will print its "Beginning search." internally.
        # It also handles its own initial "Running nearest neighbor with all X features..." print.
        algorithm_name = "Backward Elimination"
        best_features, best_accuracy, plot_data_results = backward_elimination(my_dataset)
    else:
        print("Invalid choice. Please enter 1 or 2.")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # --- Print Running Time ---
    print(f"\nAlgorithm finished in {elapsed_time:.2f} seconds.")

    import os

    print_machine_info()

    dataset_base_name = os.path.basename(dataset_file)
    plot_accuracy_vs_features(plot_data_results, algorithm_name, dataset_base_name)

