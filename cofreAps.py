import tkinter as tk

file = open("output.txt", "r")
content = file.read()

if content == "Funcionario":
    perms = " Acesso de nivel baixo permitido "
elif content == "Diretor":
    perms = " Acesso de nivel medio permitido "
elif content == "Ministro":
    perms = " Acesso de nivel alto permitido "
else:
    perms = " Acesso negado "

print(perms)

root = tk.Tk()
label = tk.Label(root, text=perms , font=("Times", 24), fg="black", bg="white")
label.pack()
root.mainloop()