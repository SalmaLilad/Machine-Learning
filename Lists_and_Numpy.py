#lists
exam1 = 85
exam2 = 93
exam3 = 88
exam_total = exam1 + exam2 + exam3
average = exam_total / 3
print(average)

# tem is temperature, in fahrenheit
tem1 = 70
tem2 = 73
tem3 = 68
tem4 = 71
tem5 = 73
tem6 = 80
tem7 = 110
tem_week= tem1+tem2+tem3+tem4+tem5+tem6+tem7
average_tem = tem_week/7
print(average_tem)

exam_scores = [85, 93, 88]
exam_total = sum(exam_scores)
average = exam_total / 3
print(average)

tem_week = [70, 73, 68, 71, 73, 80, 78, 90, 92, 110]
tem_total = sum(tem_week)
average_tem = tem_total/len(tem_week)
print(average_tem)

print(len(tem_week))

exam_scores = [85, 93, 88]
exam_total = sum(exam_scores)
average = exam_total / len(exam_scores)
print(average)

turtle_names = ["Leonardo", "Michelangelo", "Donatello", "Raphael"]
print("Ninja Turtles:") #This is not part of the loop
for name in turtle_names:
  print(name)

data = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
for row in data:
  print(row)

#numpy
import numpy as np

our_array = np.array([1,2,3,4,5])
print(our_array)

data = np.array([238, 458, 237, 549, 327, 4589])

print("Average value:")
print(np.average(data))

print("Median value:")
print(np.median(data))

print("Variance:")
print(np.var(data))

print("Standard deviation:")
print(np.std(data))

#Exercises
import numpy as np
arr1 = np.array([503,628,429,109,720,624,598])
total_income = np.sum(arr1)
print(total_income)

# prompt: The next week, your income on each day was $503, $628, $429, $109, $720, $624, and $598. Your expenses each day were $305, $352, $299, $372, $302, $340, and $333. Write a program that stores these values in a list, computes your total profits for the week.

import numpy as np

# Create lists for income and expenses
income = np.array([503, 628, 429, 109, 720, 624, 598])
expenses = np.array([305, 352, 299, 372, 302, 340, 333])

# Calculate total profits by subtracting expenses from income
total_profits = np.sum(income) - np.sum(expenses)

# Print the total profits
print("Total profits for the week:", total_profits)

# prompt: Consider the week in which your income on each day was $503, $628, $429, $109, $720, $624, and $598. Write a program that stores these values in a numpy array, and computes your average daily income this week.

import numpy as np

# Create a numpy array with the income values
income = np.array([503, 628, 429, 109, 720, 624, 598])

# Calculate the average daily income
average_income = np.average(income)

# Print the average daily income
print("Average daily income:", average_income)

# prompt: Consider the week in which your income on each day was $503, $628, $429, $109, $720, $624, and $598. Write a program to compute the variance of your daily income this week.

import numpy as np

# Create a numpy array with the income values
income = np.array([503, 628, 429, 109, 720, 624, 598])

# Calculate the variance of the income
income_variance = np.var(income)

# Print the income variance
print("Income variance:", income_variance)
