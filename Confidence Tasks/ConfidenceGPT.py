import os
from dotenv import load_dotenv
import pandas as pd
import math
from openai import OpenAI

# Loading access token
load_dotenv()
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN_GPT')

#Loading dataset of questions
generalQA = pd.read_csv('../Data/Processed/generalQA.csv', sep = '\t')

# Taking every 5th element to reduce size (and not take from only one category)
generalQA = generalQA.iloc[::5]

# Tracking correctness
correct = 0
incorrect = 0

#Establishing ChatGPT client
client = OpenAI(api_key = ACCESS_TOKEN)

# Opening output file
fWrite = open("../Outputs/outputGPT.txt", "w")

# Iterate through questions
for i in range(len(generalQA)):
    # Displaying progress
    if (i % 50 == 0):
        print (i / len(generalQA))
    # # Reading a specific question
    index = i * 5
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
    Select one from the following list and return only the number: {optionsString}
    """

    # Call the API, requesting logprobs and 10 top_logprobs
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[dict(role="user", content=prompt)],
        max_tokens=1,
        logprobs=True,
    )

    # Extract the options and confidences
    choice = completion.choices[0]
    confidence = math.exp(choice.logprobs.content[0].logprob)

    # Check correctness
    correctness = 0
    if choice.message.content == str(answer):
        correctness = 1
        correct += 1
    else:
        correctness = 0
        incorrect += 1

    # Write to output
    fWrite.write(str(index) + ": " + str(correctness) + " " + str(confidence) + " " +"\n")

fWrite.close()
print(correct, " correct answers, ", incorrect, " incorrect answers")