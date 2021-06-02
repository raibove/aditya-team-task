#Shweta Kale TECOA145
import numpy as np
import random
words = ['angel', 'race', 'pillow','ínsane','life','happy','moments','quality','time','precious','people']
word = random.choice(words)
turns = 12
guess = " "
new = word

word1=[]
word2=[]
name = input(print("Enter your name: "))
print("Good Luck!",name)
print("Guess the characters")
for l in word:
	print("-")
	word1.append(l)
	word2.append("-")



while(turns):
	guess = input("Guess a charecter: ")
	for index in range(len(word1)):
		if guess == word1[index]:
			word2[index] = guess
	for letter in word2:
		print(letter)
	turns = turns-1
	if(word1==word2):
		break

if(word1==word2):
	print("You Win")
	print(":-)")
else:
	print("You Lose")
	print(":-(")
print("The word is:",new)