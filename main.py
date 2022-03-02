# IMPORT STATEMENTS
# IO libraries
import os
import sys

# Our Modules
import model as m

# MAIN METHOD
# Unzip the data
print('UNZIPPING THE DATA')
print('Unzipping the training data')
os.system('python data.py')

print('Unzipping the test data')
os.system('python data.py --test 1')

if len(sys.argv) > 1:
    if sys.argv[1] == 'build':
        m.build_models()
    elif sys.argv[1] == 'run':
        m.get_results()
else:
    m.build_models_and_print_results()

# Run Models
