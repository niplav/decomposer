import metaculus

from datetime import datetime
from openai import OpenAI
from os.path import exists

system_prompt='''
You are an expert forecaster, skilled giving accurate probabilities
about future events. You are direct, precise and  quantitative. If
someone asks you for a number for something, you give it straight away.
'''

direct_estimation_prompt='''
Provide your best probabilistic estimate for the following question.
Give ONLY the probability, no other words or explanation. For example:
0.1
Give the most likely guess, as short as possible; not a complete sentence,
just the guess!

The question is: ${QUESTION}.
${RESOLUTION_CRITERIA}.
'''

multiplicative_decomposition_prompt=\
'''Provide your best probabilistic estimate for the following question:

${QUESTION}

Your output should be structured in three parts.

First, determine a list of factors X₁, …, X_n that are necessary
and sufficient for the question to be answered 'Yes'. You can choose
any number of factors.

Second, for each factor X_i, estimate and output the conditional
probability P(X_i|X₁, X₂, …, X_{i-1}), the probability that X_i
will happen, given all the previous factors *have* happened. Then, arrive
at the probability for Q by multiplying the conditional probabilities
P(X_i):

P(Q)=P(X₁)*P(X₂|X₁)…P(X_n|X₁, X₂, …, X_{n-1}).

Third and finally, In the last line, report P(Q), WITHOUT ANY ADDITIONAL
TEXT. Just write the probability, and nothing else.

Example (Question: 'Will my wife get bread from the bakery today?'):

Necessary factors:
1. My wife remembers to get bread from the bakery.
2. The car isn't broken.
3. The bakery is open.
4. The bakery still has bread.

1. P(My wife remembers to get bread from the bakery)=0.75
2. P(The car isn't broken|My wife remembers to get bread from the bakery)=0.99
3. P(The bakery is open|The car isn't broken, My wife remembers to get bread from the bakery)=0.7
4. P(The bakery still has bread|The bakery is open, The car isn't broken, My wife remembers to get bread from the bakery)=0.9
Multiplying out the probabilities: 0.75*0.99*0.7*0.9=0.467775
0.467775
(End of output)
Reminder, the question is: ${QUESTION}.
${RESOLUTION_CRITERIA}
'''

api_key_file=open('./.env', 'r')
key=api_key_file.read()

client = OpenAI(api_key=key)

questions=metaculus.load_questions(data_dir='../iqisa/data')
end_training_data=datetime.fromisoformat('2023-04-30')
not_in_training_data=questions.loc[(questions['q_status']=='resolved') & (questions['resolve_time']>=end_training_data)]

i=0
limit=50

for index, question in not_in_training_data.iterrows():
	direct_file_name='completions/'+str(question['question_id'])+'_direct_completion'
	multiplicative_file_name='completions/'+str(question['question_id'])+'_multiplicative_completion'
	if exists(direct_file_name) and exists(multiplicative_file_name):
		continue

	print(question['q_title'])
	print(question['resolution_criteria'])
	print('----------------------------')
	direct_prompt=direct_estimation_prompt.replace('${QUESTION}', question['q_title']).replace('${RESOLUTION_CRITERIA}', question['resolution_criteria'])
	multiplicative_prompt=multiplicative_decomposition_prompt.replace('${QUESTION}', question['q_title']).replace('${RESOLUTION_CRITERIA}', question['resolution_criteria'])

	direct_completion = client.chat.completions.create(
	  model='gpt-4',
	  messages=[
	    {'role': 'system', 'content': system_prompt},
	    {'role': 'user', 'content': direct_prompt}
	  ]
	)
	multiplicative_completion = client.chat.completions.create(
	  model='gpt-4',
	  messages=[
	    {'role': 'system', 'content': system_prompt},
	    {'role': 'user', 'content': multiplicative_prompt}
	  ]
	)
	print(direct_completion)
	print(multiplicative_completion)
	print('==================================')

	direct_completion_file=open(direct_file_name, 'w+')
	direct_completion_file.write(direct_completion.choices[0].message.content)
	direct_completion_file.close()

	multiplicative_completion_file=open(multiplicative_file_name, 'w+')
	multiplicative_completion_file.write(multiplicative_completion.choices[0].message.content)
	multiplicative_completion_file.close()

	i=i+1

	if i>=limit:
		break
