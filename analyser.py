import os
import re
import metaculus

import pandas as pd
import numpy as np

from datetime import datetime

def logscore(o,p):
	return o*np.log(p)+(1-o)*np.log(1-p)

questions=metaculus.load_questions(data_dir='../iqisa/data')
end_training_data=datetime.fromisoformat('2023-04-30')
not_in_training_data=questions.loc[(questions['q_status']=='resolved') & (questions['resolve_time']>=end_training_data)]

direct_forecasts=dict()
multiplicative_forecasts=dict()

# Regular expression to match floating point numbers
float_regex = re.compile(r'^-?\d+(?:\.\d+)?$')

# Path to the directory
dir_path = "completions/"

# Iterate over each file in the directory
for file_name in os.listdir(dir_path):
	# Construct full file path
	file_path = os.path.join(dir_path, file_name)

	# Initialize variable to hold the last found float value
	last_float = None

	# Open and read the file
	with open(file_path, 'r') as file:
		for line in file:
			# Strip whitespace from the line
			line = line.strip()
			# Check if the line is a floating point number
			if float_regex.match(line):
				last_float = float(line)

	# If a floating point number was found, determine the dictionary and key
	if last_float is not None:
		# Extract the key (number at the start of the file name) using regex
		key = int(re.match(r'\d+', file_name).group())

		# Check if it's a direct or multiplicative completion
		if "direct_completion" in file_name:
			direct_forecasts[key] = last_float
		elif "multiplicative_completion" in file_name:
			multiplicative_forecasts[key] = last_float

selected_outcomes = not_in_training_data[not_in_training_data['question_id'].isin(direct_forecasts.keys())][['question_id', 'outcome']]
selected_outcomes['outcome']=selected_outcomes['outcome'].astype(np.float64)

direct_forecasts_df = pd.DataFrame(list(direct_forecasts.items()), columns=['question_id', 'forecast'])
direct_forecasts_df=direct_forecasts_df.sort_values(by='question_id')

multiplicative_forecasts_df = pd.DataFrame(list(multiplicative_forecasts.items()), columns=['question_id', 'forecast'])
multiplicative_forecasts_df=multiplicative_forecasts_df.sort_values(by='question_id')

print(np.mean(logscore(np.array(selected_outcomes['outcome']), np.array(direct_forecasts_df['forecast']))))
print(np.mean(logscore(np.array(selected_outcomes['outcome']), np.array(multiplicative_forecasts_df['forecast']))))
