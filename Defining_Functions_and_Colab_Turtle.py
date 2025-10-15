def average(first_number, second_number, third_number):
  total = first_number + second_number + third_number
  average = total / 3
  return average

result = average(38, 28, 459)
print(result)

def average(numbers):
    sum = 0
    for value in numbers:
        sum = sum + value
    return sum / (len(numbers))

exam_scores = [85, 93, 88]
average_score = average(exam_scores)
print(average_score)

!pip install ColabTurtle

from ColabTurtle import Turtle

def draw_triangle(side_length):
  for edge in range(3):
    Turtle.forward(side_length)
    Turtle.right(120)

Turtle.initializeTurtle()
draw_triangle(50)
draw_triangle(100)
draw_triangle(150)

Turtle.initializeTurtle()

for branch in range(6):
  for length in [50,100,150,200]:
    draw_triangle(length)
  Turtle.right(60)
