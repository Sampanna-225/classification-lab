with open("file.txt" , "w") as f:
    f.write("This is a file")

with open("file.txt" , "r") as f:
    content = f.read()
    print(content)