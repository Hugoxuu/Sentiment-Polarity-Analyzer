import sys
import numpy as np
import math

# Command line arguments
train_input = sys.argv[1]  # path to the training input .tsv file
valid_input = sys.argv[2]  # path to the validation input .tsv file
test_input = sys.argv[3]  # path to the test input .tsv file
dict_input = sys.argv[4]  # path to the dictionary input .txt file
train_out = sys.argv[5]  # path to the training output .labels file
test_out = sys.argv[6]  # path to the test output .labels file
metrics_out = sys.argv[7]  # path to the output .txt file to which metrics like train and test error should be written
num_epoch = int(sys.argv[8])  # integer specifying the number of times SGD loops

# global variable
learning_rate = 0.1
feature_value = 1  # The feature value is always 1 in this case


# efficient dot product for logistic regression
def sparse_dot(X, W):  # X as sample array in dict type, W as feature array
    product = 0.0
    for k in X:
        product += W[k]  # The feature value is always 1, so skipped in the equation
    return product


# Output label (y) and design (x) matrix and number of samples from given data
def data_process(data_input):
    # Get the number of samples
    with open(data_input, 'r') as data:
        num_sample = sum(1 for line in data)

    # Get label(y) and design(x) matrix from given data
    with open(data_input, 'r') as data:
        y = np.zeros(num_sample)  # N x 1
        x = list()  # N x 1-(dict), every dict representing each sample
        line_num = 0
        for line in data:
            sample = np.array(line.split(sep='\t'))  # take a sample into a list
            y[line_num] = sample[0]  # y(i) = label
            x_i = dict()  # create dict type for every new sample
            x_i[0] = 1  # always include the bias = 1 for each sample
            sample = np.delete(sample, 0)   # update the sample matrix (get rid of the label data)
            for ele in sample:
                dict_part = ele.split(sep=':')
                feature_index = int(dict_part[0]) + 1  # plus 1 for every index because of the bias term
                x_i[feature_index] = feature_value  # x(i) = feature value: 1 (the feature_value is always 1)
            x.append(x_i)
            line_num += 1  # for updating index of y,x
    return y, x, num_sample


# Output the prediction of certain data (x)
def predictor(data_output, model_feature, x, num_sample):
    with open(data_output, 'w') as res:
        i = 0
        while i < num_sample:
            feature_times_x = sparse_dot(x[i], model_feature)
            prob = 1 / (1 + math.exp(feature_times_x))
            result = 0 if prob > 0.5 else 1
            print(result, file=res)
            i += 1
    return


# Main Method #

# Get the number of features in dict.txt
with open(dict_input, 'r') as file:
    num_feature = sum(1 for line in file)  # get the number of features

# Initialize feature matrix
feature = np.zeros(num_feature + 1)  # dimension is (#feature + 1) because of the bias term

# Get the label (y) and design (x) matrix from train data
(y_train, x_train, train_num_sample) = data_process(train_input)

# Start training 'feature' variable using SGD
i = 0  # the i-th sample
iteration = num_epoch  # num of loops for iteration
while iteration > 0:  # loop num_epoch times
    while i < train_num_sample:  # loop num_sample times
        feature_update = np.zeros(num_feature + 1)  # SGD update for feature
        feature_times_x = sparse_dot(x_train[i], feature)
        scalar = y_train[i] - (math.exp(feature_times_x) / (1 + math.exp(feature_times_x)))
        for index in x_train[i]:
            feature_update[index] = scalar
        feature += learning_rate * feature_update  # feature update
        i += 1  # continue with the next sample
    i = 0
    iteration -= 1  # continue with the next iteration

# Train out
predictor(train_out, feature, x_train, train_num_sample)

# Get the label (y) and design (x) matrix from test data
(y_test, x_test, test_num_sample) = data_process(test_input)

# Test out
predictor(test_out, feature, x_test, test_num_sample)

# metrics out
with open(metrics_out, 'w') as res:
    train_mismatch = 0
    test_mismatch = 0
    with open(train_out, 'r') as pred:
        i = 0
        for line in pred:
            if y_train[i] != int(line):
                train_mismatch += 1
            i += 1
    with open(test_out, 'r') as pred:
        i = 0
        for line in pred:
            if y_test[i] != int(line):
                test_mismatch += 1
            i += 1
    print("error(train): %.6f" % (train_mismatch / train_num_sample), file=res)
    print("error(test): %.6f" % (test_mismatch / test_num_sample), file=res)
