from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import tkinter as tk
from test import modelgencptn
#model_name , photo_name
class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Project")
        self.minsize(640, 400)

        self.labelFrame = ttk.LabelFrame(self, text = "Open File")
        self.labelFrame.grid(column = 0, row = 1, padx = 20, pady = 20)

        self.button()
        B = ttk.Button(self.labelFrame ,text = "Submit" , command = self.abc )
        B.grid(column=1, row = 1000)



    def abc(self):
        model_name = './model_9.h5'
        photo_name = self.filename
        Cap = modelgencptn(model_name , photo_name)
        Cap = Cap.split()[1:-1]
        Cap = ' '.join(Cap)
        label = tk.Label(text=Cap, fg="white", bg="black")
        label.grid(column=1,row=2000)
        label.config(font=("Courier", 14))
        

    def button(self):
        self.button = ttk.Button(self.labelFrame, text = "Browse A File",command = self.fileDialog)
        self.button.grid(column = 1, row = 1)


    def fileDialog(self):

        self.filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =
        (("jpeg files","*.jpg"),("all files","*.*")) )
        self.label = ttk.Label(self.labelFrame, text = "")
        self.label.grid(column = 1, row = 2)
        self.label.configure(text = self.filename)

        img = Image.open(self.filename)
        img = img.resize((224,224) , Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)

        self.label2 = Label(image=photo)
        self.label2.image = photo 
        self.label2.grid(column=1, row=4)

    

root = Root()
root.mainloop()