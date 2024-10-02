import os
from dotenv import load_dotenv
import torch
from transformers import pipeline
import numpy as np
import pandas as pd
import re

# Loading access token
load_dotenv()
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN_HF')

#Loading dataset of questions
generalQA = pd.read_csv('../Data/Processed/generalQA.csv', sep = '\t')

# Tracking correctness
correct = 0
incorrect = 0

# Opening output file
fWrite = open("../Outputs/verbalizationOutputLlama.txt", "w")

# Setting up model
model_name = "meta-llama/Llama-2-7b-chat-hf"

generator = pipeline("text-generation", model_name, 
    token = ACCESS_TOKEN, 
    device = "cuda",  
    temperature=0.1,
    repetition_penalty=1.1)

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
    prompt = f"""\<s>[INST] <<SYS>>
    Read the question, provide your best guess and the probability that it is correct (0.0 to 1.0) for
    the following question. Give ONLY the guess and probability, no other words or
    explanation. For example:\n\nGuess: <most likely guess, as short as possible; not
    a complete sentence, just the option number!>\n Probability: <the probability between 0.0
    and 1.0 that your guess is correct, without any extra commentary whatsoever; just
    the probability!>
    <</SYS>>
    The question is: {questionString}
    Select one from the following list and return only the number: {optionsString}
    """
    output = generator(prompt, max_new_tokens = 25)[0]['generated_text'][len(prompt):]

    # Split message
    output_lines = output.split("\n") 
    # Check if output matches format (currently, very strict, could be loosened)
    if len(output_lines) >= 2 and \
        output_lines[0].strip().startswith("Guess:") and \
        output_lines[1].strip().startswith("Probability: "):
        # Parse out choice and confidence
        choices = re.findall("\d+", output_lines[0])
        confidences = re.findall("\d+\.?\d+", output_lines[1])
        if (len(choices) == 1 and len(confidences) == 1):
            choice = int(choices[0])
            confidence = float(confidences[0])

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
            continue

    # Output was skipped due to poor generation (TODO: perhaps try multiple times?)
    fWrite.write(str(index) + ": " + "skipped \n")

# Close file and print outcome
fWrite.close()
print(str(correct) + " correct answers, " + str(incorrect) + " incorrect answers")
