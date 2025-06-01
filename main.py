import math
import time   

# Instance class for creating an instance of particular sample/row from the dataset to be processed
class Instance:

    def __init__(self, features, class_label): # this is the constructor method for creating new instance.
        self.features = features # setting the features we pass as instance features
        self.class_label = class_label # same for the class label

    def __repr__(self): # this is method I have defined for printing a particular instance with all it's features and class label.
        return f"Instance(features={self.features}, label={self.class_label})"
    
# this is a function to load the dataset from the file
def load_dataset(file_path):

    dataset = [] # list to store all the instances

    try:    # open the file, check if the file is appropriate and 
            #then take each line in file, do some computations and store 1st value as class label and all other as features.
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
    # if file is not appropriate, throw an error to let the user know
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{file_path}'. Please check the path.")
        return [] 
    except Exception as e:
        print(f"An unexpected error occurred while loading '{file_path}': {e}")
        return []

# This is a function to calculate the euclidean distance between two instances which we will use later on for finding the nearest neighbor
def euclidean_distance(instance1_features, instance2_features):
   
    # just an initial check if both have same number of features as without it we can't really find the euclidean distance between both of them 
    if len(instance1_features) != len(instance2_features):
        raise ValueError("Feature vectors must have the same number of dimensions for Euclidean distance calculation.")

    sum_sq_diff = 0 # storing final ans.

    for i in range(len(instance1_features)): # looping through all features of both instances, finding there difference and adding the square of them
        difference = instance1_features[i] - instance2_features[i]
        sum_sq_diff += (difference ** 2) 

    return math.sqrt(sum_sq_diff) # returning square root of final ans, which shows properly that we are taking proper euclidean disatnce between two instances

# This function is to finally calculate the accuracy of our nearest neighbor algorithm as mentioned in project description with Leave-One-Out-Cross-Validation
def nearest_neighbor_accuracy(dataset, current_feature_indices):
    
    # check if we have a proper dataset to train and test on.
    if not dataset:
        print("Warning: Dataset is empty. Cannot calculate accuracy. Returning 0%.")
        return 0.0
    if not current_feature_indices:
        print("Warning: No features selected for accuracy calculation. Returning 0%.")
        return 0.0

    num_correct_predictions = 0 # storing correct predictions
    for i in range(len(dataset)): # loops over all the instances and the ith instance is the test instance
        test_instance = dataset[i]

        test_features_subset = [test_instance.features[idx] for idx in current_feature_indices]

        min_distance = float('inf') # considering k = 1 by default as hard-coded value
        nearest_neighbor_class = -1 

        for j in range(len(dataset)):   # looping over all instance other than test and 
                                        # finding the distance of test with all other instances and storing the minimum distance and it's label
            if i == j:
                continue 

            training_instance = dataset[j] 

            training_features_subset = [training_instance.features[idx] for idx in current_feature_indices]

            distance = euclidean_distance(test_features_subset, training_features_subset)

            if distance < min_distance:
                min_distance = distance 
                nearest_neighbor_class = training_instance.class_label 

        if nearest_neighbor_class == test_instance.class_label: # we check that the label we got from our algorithm matches actual one and if it does we increment correct predictions
            num_correct_predictions += 1 

    accuracy = (num_correct_predictions / len(dataset)) * 100 # calculate the accuracy and return it
    return accuracy

# implemented forward selection that starts with no feature, adds one feature in each step that improves accuracy most and continues until all features are added or no improvement is seen.
def forward_selection(dataset):

    num_total_features = len(dataset[0].features) 
    
    current_features = set() 
    best_overall_features = set() 
    best_overall_accuracy = -1.0 

    print("\nBeginning Forward Selection Search.\n")

    for i in range(num_total_features):
        print(f"  On level {i + 1} of the search tree, considering adding a feature to {sorted([f + 1 for f in current_features])}...")

        feature_to_add_this_level = -1
        best_accuracy_this_level = -1.0
        
        for candidate_feature_idx in range(num_total_features):
            if candidate_feature_idx not in current_features:
                temp_features = current_features.union({candidate_feature_idx})
                
                accuracy = nearest_neighbor_accuracy(dataset, temp_features)

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

        if best_accuracy_this_level > best_overall_accuracy:
            best_overall_accuracy = best_accuracy_this_level
            best_overall_features = current_features.copy()
        else:
            print(f"(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
            print(f"(Current best feature subset: {set(sorted([f + 1 for f in best_overall_features]))}, accuracy: {best_overall_accuracy:.2f}%)\n")

    print(f"Finished search!! The best feature subset is {set(sorted([f + 1 for f in best_overall_features]))}, which has an accuracy of {best_overall_accuracy:.2f}%")
    return best_overall_features, best_overall_accuracy

# implemented backward selection
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

# This is the main fucntion that is executed.
def main() :

    print("Welcome to Akshat Shah's Feature Selection Algorithm.\n")

    # --- 2. Ask for File Name ---
    dataset_file = input("Type in the name of the file to test : ")
    print()

    my_dataset = load_dataset(dataset_file) # converting the dataset file into dataset of instances.

    while not my_dataset: # Loop until a valid, non-empty dataset is loaded
      dataset_file = input("Type in the name of the file to test : ")
      my_dataset = load_dataset(dataset_file)

    num_instances = len(my_dataset) # number of samples in dataset 

    num_features = len(my_dataset[0].features) if num_instances > 0 else 0 # number of features in our dataset

    print(f"This dataset has {num_features} features (not including the class attribute), with {num_instances} instances.\n")

    all_feature_indices = set(range(num_features))
    
    initial_all_features_accuracy = nearest_neighbor_accuracy(my_dataset, all_feature_indices)
    print(f"Running nearest neighbor with all {num_features} features, using \"leaving-one-out\" evaluation, I get an accuracy of {initial_all_features_accuracy:.2f}%\n")
    # asking to choose the search method
    choice = ''
    while choice not in ['1', '2']:
        print("\nType the number of the algorithm you want to run.")
        print("1) Forward Selection")
        print("2) Backward Elimination")
        choice = input("Enter choice (1 or 2): ")
        if choice not in ['1', '2']:
            print("Invalid choice. Please enter 1 or 2.")

    start_time = time.time()
    # running the algorithm based on chosed method
    if choice == '1':
        algorithm_name = "Forward Selection"
        best_features, best_accuracy, plot_data_results = forward_selection(my_dataset, initial_all_features_accuracy)
    elif choice == '2':

        algorithm_name = "Backward Elimination"
        best_features, best_accuracy, plot_data_results = backward_elimination(my_dataset)
    else:
        print("Invalid choice. Please enter 1 or 2.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    #finally showing the time taken to run the algorithm
    print(f"\nAlgorithm finished in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
