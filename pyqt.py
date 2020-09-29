import turtle
from tkinter import *
 
def drawp():
    theScreen = turtle.TurtleScreen(canva)
    path = turtle.RawTurtle(theScreen)
    path.forward(100)
    path.right(90)
    path.forward(150)
    theScreen.mainloop()
 
root = Tk()
root.title("the path")
 
canva = Canvas(root, width=400, height=400)
canva.pack()
 
aa = Button(root, text="Draw", command=drawp)
aa.pack()
 
root.mainloop()
