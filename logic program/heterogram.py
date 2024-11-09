
word = input("Enter a word: ")

word = word.lower()

is_heterogram = True

for i in range(len(word)):
        for j in range(i + 1, len(word)):
            if word[i] == word[j]:
            
                is_heterogram = False
                break
        if not is_heterogram:
            break


if is_heterogram:
    print("The word is a heterogram.")
else:
    print("The word is not a heterogram.")
