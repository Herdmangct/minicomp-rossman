# IO libraries
import sys

# Our Modules
import model as m

# Get user input 
test_data_path = sys.argv[1]

# run the model
m.build_models_and_print_results(test_data_path=test_data_path)