n = int(input("Enter the number of terms for the Fibonacci series: "))
a, b = 0, 1
print("Fibonacci series:")
for i in range(n):
    print(a, end=" ")
    a, b = b, a + b
