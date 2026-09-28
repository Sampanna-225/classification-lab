shopping_list = ["Gas"]
shopping_list.append("Milk")
j=1
def check():
    if "Milk" in shopping_list:
        print("There is Milk")
    else:
        print("No Milk")

check()
shopping_list.remove("Milk")
check()

print(len(shopping_list))
for i in shopping_list:
    print(j , ".",i , " ")
    j+=1
