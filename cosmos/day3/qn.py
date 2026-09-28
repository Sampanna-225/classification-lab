note = input("Enter a note: ")
file = input("File Name : ")
try:
    with open(file , "w") as f:
        f.write(note)
except Exception as e:
    print(e)

try:
    with open(file , "r") as f:
        con = f.read()
        print(con)
except FileNotFoundError:
    print("File not found")
