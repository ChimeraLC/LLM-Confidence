import numpy as np
import matplotlib.pyplot as plt
# Read file
fRead = open("output.txt", "r")

# Bin graph values
binTotals = [0,0,0,0,0, 0] #0, 0.2, 0.4, 0.6, 0.8
binCorrects = [0, 0, 0, 0, 0, 0]
overallTotal = 0

# Read correctness and confidence values
correctness = []
confidences = []
while line := fRead.readline():
    elements = line.split()
    correctness.append(float(elements[1]))
    confidences.append(float(elements[2]))
    # Track totals for each bin
    binTotals[int(float(elements[2]) * 5)] += 1
    binCorrects[int(float(elements[2]) * 5)] += int(elements[1])
    overallTotal += 1

# Fold last option back onto itself
binTotals[4] += binTotals.pop(5)
binCorrects[4] += binCorrects.pop(5)
binRatios = []
for i in range(5):
    binRatios.append(binCorrects[i]/binTotals[i])

print(np.corrcoef(correctness, confidences))
# Plot values
import matplotlib.pyplot as plt

bins = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
 
fig = plt.figure(figsize = (10, 5))

# Generate colors for density
colors = []
for i in range(5):
    ratio = binTotals[i]/overallTotal
    norm_ratio = np.power(1 - ratio, 6)
    colors.append((norm_ratio, norm_ratio, norm_ratio))

# creating the bar plot
plt.bar(bins, binRatios, color =colors, 
        width = 1.0)
plt.ylim(0, 1)
plt.xlabel("Confidences")
plt.ylabel("Percentage answered correct")
plt.title("Reliability graph of confidences")
plt.show()
plt.savefig("correlation.png")
