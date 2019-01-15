import sys
import csv

# Command line arguments
train_input = sys.argv[1]  # path to the training input .tsv file
valid_input = sys.argv[2]  # path to the validation input .tsv file
test_input = sys.argv[3]  # path to the test input .tsv file
dict_input = sys.argv[4]  # path to the dictionary input .txt file
train_out = sys.argv[5]  # path to the training output .tsv file
valid_out = sys.argv[6]  # path to the validation output .tsv file
test_out = sys.argv[7]  # path to the test output .tsv file
model = int(sys.argv[8])  # integer taking value 1 or 2 specifying whether to use Model 1 or Model 2 feature set

# global variables
names = [[train_input, train_out], [valid_input, valid_out], [test_input, test_out]]
trim_threshold = 4

# Transfer the dict.txt into python dictionary
with open(dict_input, 'r') as file:
    dict = {}
    for line in file:
        (word, index) = line.split()
        dict[word] = int(index)

# feature flag selection
if model == 1:  # model 1
    for name in names:  # input and output simultaneous
        with open(name[0], 'r') as tsvin, open(name[1], 'w') as tsvout:
            tsvin = csv.reader(tsvin, delimiter='\t')
            for line in tsvin:
                label = line[0]
                words = line[1]
                word = words.split()  # Get all the words in one certain review into a list
                word_set = set()  # use set to get rid of the duplicates in the "word" list
                for ele in word:  # ...
                    word_set.add(ele)  # ...
                row = list();  # the output to be written on one line of the tsv output file
                row.append(str(label))
                for ele in word_set:
                    if dict.get(ele) is not None:
                        temp = str(dict[ele]) + ":1"
                        row.append(temp)
                print('\t'.join(row), file=tsvout)  # write to tsvout
else:  # model 2
    for name in names:  # input and output simultaneous
        with open(name[0], 'r') as tsvin, open(name[1], 'w') as tsvout:
            tsvin = csv.reader(tsvin, delimiter='\t')
            for line in tsvin:
                label = line[0]
                words = line[1]
                word = words.split()  # Get all the words in one certain review into a list
                word_dict = {}  # use dictionary to count the occurrence of words in the list
                for ele in word:  # ...
                    if word_dict.get(ele) is None:  # ...
                        word_dict[ele] = 1  # ...
                    else:   # ...
                        word_dict[ele] = word_dict.get(ele) + 1  # ...
                row = list()  # the output to be written on one line of the tsv output file
                row.append(str(label))
                for ele in word_dict:
                    if (dict.get(ele) is not None) and (word_dict[ele] < trim_threshold):
                        temp = str(dict[ele]) + ":1"
                        row.append(temp)
                print('\t'.join(row), file=tsvout)  # write to tsvout