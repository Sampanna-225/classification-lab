def save_note():
    note = input("Type a note to save: ")
    with open("notes.txt", "w") as f:
        f.write(note)
    print("Note saved in notes.txt")

def read_file_by_name():
    filename = input("Enter a filename to read: ")
    try:
        with open(filename , "r") as f:
            print(f.read())
    except FileNotFoundError:
        print("Sorry File name doesnt exist")

def repeat_note():
    note = input("Type a note: ")
    try:
        times = int(input("How many number of times : "))
        with open("repeated_notes.txt", "w") as f:
            for i in range(times):
                f.write(note + "\n")
        print("Saved in repeated nores")
    except ValueError:
        print("Enter valid number ")

save_note()
read_file_by_name()
repeat_note()