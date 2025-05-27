import math

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

# This is the main fucntion that is executed.
def main() :
    dataset_file = 'CS205_small_Data__11.txt' # path to the input dataset file

    my_dataset = load_dataset(dataset_file) # converting the dataset file into dataset of instances.

    if not my_dataset:
        exit()

    num_instances = len(my_dataset) # number of samples in dataset 

    num_features = len(my_dataset[0].features) if num_instances > 0 else 0 # number of features in our dataset

    print(f"This dataset has {num_features} features (not including the class attribute), with {num_instances} instances.")

    all_feature_indices = set(range(num_features))
    
    accuracy_all_features = nearest_neighbor_accuracy(my_dataset, all_feature_indices)
    print(f"Running nearest neighbor with all {num_features} features, using \"leaving-one-out\" evaluation, I get an accuracy of {accuracy_all_features:.1f}%")

if __name__ == "__main__":
    main()
