# IMPORT STATEMENTS
# IO libraries
import os

# Our Modules
import model as m

# MAIN METHOD
# Install the required packages
print('INSTALLING THE REQUIRED PACKAGES')
os.system('pip install pandas')
os.system('pip install numpy')
os.system('pip install scikit-learn')
os.system('pip install xgboost')
# os.system('pip install -r requirements.txt')
print()

# Unzip the data
print('UNZIPPING THE DATA')
print('Unzipping the training data')
os.system('python data.py')

print('Unzipping the test data')
os.system('python data.py --test 1')

# Run the model
m.build_models_and_print_results(test_data_path='./data/holdout.csv')