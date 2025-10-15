#play against a computer -> computer always wins
import time

!wget -q http://www-users.math.umn.edu/~jwcalder/NimMaster.py
import NimMaster

n = 20 #Number of objects in the Nim pile
your_turn = True #Do you go first, or NimMaster?

print("Nim: Starting with %d items.\n"%n)

#Play the game
while(n >= 2):  #While there are at least 2 objects left

  if your_turn:
    your_play = input("Enter your play:")
    your_play = max(min(int(your_play),min(3,n)),1) #Make sure the play is 1,2, or 3
    print("You removed %d item(s)"%your_play)
    n = n - your_play
    if(n >= 1):
      your_turn = False
  else: #It is NimMaster's turn
    time.sleep(1)
    NimMaster_play = NimMaster.play(n)
    print("NimMaster removed %d item(s)"%NimMaster_play)
    n = n - NimMaster_play
    your_turn = True

  #Print number of objects left
  print("\nNumber of items left = %d"%n)

if(your_turn): #If game ends on your turn, you lost
  print("\n\nYou lost:(")
else:
  print("\n\nYou Won!!!")

#making your own algorithm -> you will always win
import time
import NimMaster

def YourAlgorithm(n):
  #add your code here to determine your_play
  #For example:
  your_play = max(1, (n-1)%4)

  # if n == 4 or n == 3:
  #   your_play = 3
  # if n%3 == 1:
  #   your_play = 2
  # elif n%3 == 2:
  #   your_play = 3

  return your_play

n = 11 #Number of objects in the Nim pile
your_turn = True #Do you go first, or NimMaster?

print("Nim: Starting with %d items.\n"%n)

#Play the game
while(n >= 2):  #While there are at least 2 objects

  if your_turn:
    time.sleep(1)
    your_play = YourAlgorithm(n)
    your_play = max(min(int(your_play),min(3,n)),1) #Make sure the play is 1,2, or 3
    print("You removed %d item(s)"%your_play)
    n = n - your_play
    if(n >= 1):
      your_turn = False
  else: #It is NimMaster's turn
    time.sleep(1)
    NimMaster_play = NimMaster.play(n)
    print("NimMaster removed %d (item(s)"%NimMaster_play)
    n = n - NimMaster_play
    your_turn = True

  #Print number of objects left
  print("\nNumber of items left = %d"%n)

if(your_turn): #If game ends on your turn, you lost
  print("\n\nYou lost!")
else:
  print("\n\nYou Won!!!")
