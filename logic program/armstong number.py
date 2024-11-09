num=input("enter the armsrong number:")
sum=0
n=len(num)
for i in num:
    sum+=int(i)**n
if sum== int(num):
    print("armstrong number")
else:
    print("its not armstrong number")
