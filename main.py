# IMPORT STATEMENTS
# IO libraries
import os

# Our Modules
import model as m

# MAIN METHOD
# Unzip the data
print('UNZIPPING THE DATA')
print('Unzipping the training data')
os.system('python data.py')

print('Unzipping the test data')
os.system('python data.py --test 1')

# Run the model
m.build_models_and_print_results(test_data_path='./data/holdout.csv')