import os
from dotenv import load_dotenv
import pandas as pd
import math
from openai import OpenAI
import re

# Loading access token
load_dotenv()
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN_GPT')

#Loading dataset of questions
generalQA = pd.read_csv('../Data/Processed/generalQA.csv', sep = '\t')
generalQA = generalQA.iloc[:1000]

#Establishing ChatGPT client
client = OpenAI(api_key = ACCESS_TOKEN)

# Opening output file
fWrite = open("../Outputs/verbalizationOutput.txt", "w")

# Tracking correctness
correct = 0
incorrect = 0

# Iterate through questions
for i in range(len(generalQA)):
    # Displaying progress
    print(i / len(generalQA), end = '\r')
    # Reading a specific question
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
    Provide your best guess and the probability that it is correct (0.0 to 1.0) for
    the following question. Give ONLY the guess and probability, no other words or
    explanation. For example:\n\nGuess: <most likely guess, as short as possible; not
    a complete sentence, just the option number!>\n Probability: <the probability between 0.0
    and 1.0 that your guess is correct, without any extra commentary whatsoever; just
    the probability!>\n\nThe question is: {questionString}
    Select one from the following list and return only the number: {optionsString}
    """

    # Call the API, requesting logprobs and 10 top_logprobs
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[dict(role="user", content=prompt)],
        max_tokens=25,
        logprobs=True,
    )

    # Isolate message
    message = completion.choices[0].message.content
    message_lines = message.split("\n") 

    # Parse out choice and confidence
    choices = re.findall("\d+", message_lines[0])
    confidences = re.findall("\d+\.?\d+", message_lines[1])
    # Sanity check
    if len(choices) != 1 or len(confidences) != 1:
       print("Encountered unexpected api response, skipping to next")
       fWrite.write(str(index) + ": unexpected api response")
       continue

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
# # Call the API, requesting logprobs and 10 top_logprobs
# completion = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[dict(role="user", content=prompt)],
#     max_tokens=25,
#     logprobs=True,
# )

# #print(completion)
# print(completion.choices[0].message.content)


fWrite.close()
print(correct, "correct answers,", incorrect, "incorrect answers")