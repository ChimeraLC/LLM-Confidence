#Dataset from https://github.com/uberspot/OpenTriviaQA/blob/master/categories/world


# Opening files to write to
import csv
fWrite = open("generalQA.csv", "w")
csvwriter = csv.writer(fWrite, delimiter='\t',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
fRead = open("general.txt", "r")

# Write columns
csvwriter.writerow(["Question", "Answer", "Option1", "Option2", "Option3", "Option4"])

foundQuestion = False
foundAnswers = False
questionIndex = 0
questionString = ""
answerString = ""
correctAnswer = -1
optionStrings = ["", "", "", ""]

while line := fRead.readline():
    # End of question
    if line[0] == "\n":
        # Write question if possible
        if (foundQuestion):
            # Figure out correct answer
            for i in range(len(optionStrings)):
                if optionStrings[i] == answerString:
                    correctAnswer = i
            csvwriter.writerow([questionString, correctAnswer] + optionStrings)
        foundQuestion = False
        foundAnswers = False
        continue

    # Found new question
    if line[0] == "Q":
        foundQuestion = True
        foundAnswers = False
        questionString = line[2:-1]
        continue
    
    # Found answer
    if line[0] == "^":
        foundAnswers = True
        questionIndex = -2
        answerString = line[2:-1]

    # Found answer options
    if (foundQuestion and not foundAnswers):
        questionString += line[:-1]
    elif (foundAnswers):
        questionIndex += 1
        if (questionIndex >= 0):
            optionStrings[questionIndex] = line[2:-1]

fWrite.close()
fRead.close()