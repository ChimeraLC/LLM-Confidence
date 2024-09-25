import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd

# Loading access token
load_dotenv()
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN_HF')

#Loading dataset of questions
generalQA = pd.read_csv('../Data/Processed/generalQA.csv', sep = '\t')

# Taking every 5th element to reduce size (and not take from only one category)
generalQA = generalQA.iloc[::]

# Tracking correctness
correct = 0
incorrect = 0

# Opening output file
fWrite = open("../Outputs/outputLLama.txt", "w")

# Setting up model
model_name = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name, token = ACCESS_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_name, token = ACCESS_TOKEN)


# Iterate through questions
for i in range(len(generalQA)):
    # Displaying progress
    if (i % 50 == 0):
        print (i / len(generalQA))
    # # Reading a specific question
    index = i
    questionString = generalQA["Question"][index]
    questionOptions = [generalQA["Option1"][index], generalQA["Option2"][index], generalQA["Option3"][index], generalQA["Option4"][index]]
    answer = generalQA["Answer"][index]

    # Convert to multiple choice responses
    optionsString = "\n"
    for i, g in enumerate(questionOptions):
        if g != "":
            optionsString += f"{i}. {g}\n"

    # Form complete prompt
    prompt = f"""\
    {questionString}
    From the following options: {optionsString}
    The correct option as a number is:
    """

    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    # Send to model
    outputs = model.generate(input_ids, max_new_tokens = 3, do_sample=True, output_scores=True, return_dict_in_generate=True)

    # Calculate transition scores
    transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True
                                                        )
    # Extract the choice and probability
    input_length = input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]

    choice = -1
    confidence = 0
    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        if(tokenizer.decode(tok).isdigit()):
            choice = (int)(tokenizer.decode(tok))
            confidence = np.exp(score.numpy())
    
    # Skip over failed options
    if (choice == -1):
        continue

    # Check correctness
    correctness = 0
    if choice == answer:
        correctness = 1
        correct += 1
    else:
        correctness = 0
        incorrect += 1

    # Write to output
    fWrite.write(str(index) + ": " + str(correctness) + " " + str(confidence) + " " +"\n")

fWrite.close()
print(str(correct) + " correct answers, " + str(incorrect) + " incorrect answers")
