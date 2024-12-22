import tkinter as tk
from tkinter import colorchooser

class DrawingBoard:
    def __init__(self, root):
        self.root = root
        self.root.title("Drawing Board")
        self.root.geometry("800x600")

        # Canvas setup
        self.canvas = tk.Canvas(self.root, bg="white", width=800, height=500)
        self.canvas.pack(pady=10)

        # Tool variables
        self.current_tool = "pen"
        self.start_x = None
        self.start_y = None
        self.color = "black"
        self.eraser_on = False

        # Add buttons
        self.create_tool_buttons()

        # Bind canvas events
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def create_tool_buttons(self):
        button_frame = tk.Frame(self.root)
        button_frame.pack()

        tools = [
            ("Pen", "pen"),
            ("Line", "line"),
            ("Circle", "circle"),
            ("Rectangle", "rectangle"),
            ("Triangle", "triangle"),
            ("Eraser", "eraser"),
            ("Color", "color"),
            ("Clear", "clear"),
        ]

        for text, tool in tools:
            button = tk.Button(button_frame, text=text, command=lambda t=tool: self.select_tool(t))
            button.pack(side=tk.LEFT, padx=5, pady=5)

    def select_tool(self, tool):
        if tool == "color":
            self.color = colorchooser.askcolor(color=self.color)[1]
            self.eraser_on = False
        elif tool == "clear":
            self.canvas.delete("all")
        else:
            self.current_tool = tool
            self.eraser_on = tool == "eraser"

    def start_draw(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def draw(self, event):
        if self.current_tool == "pen":
            self.canvas.create_line(self.start_x, self.start_y, event.x, event.y, fill=self.color, width=2)
            self.start_x, self.start_y = event.x, event.y
        elif self.eraser_on:
            self.canvas.create_rectangle(event.x - 10, event.y - 10, event.x + 10, event.y + 10, fill="white", outline="white")

    def reset(self, event):
        if self.current_tool in ["line", "circle", "rectangle", "triangle"]:
            if self.current_tool == "line":
                self.canvas.create_line(self.start_x, self.start_y, event.x, event.y, fill=self.color, width=2)
            elif self.current_tool == "circle":
                r = ((event.x - self.start_x) ** 2 + (event.y - self.start_y) ** 2) ** 0.5
                self.canvas.create_oval(self.start_x - r, self.start_y - r, self.start_x + r, self.start_y + r, outline=self.color, width=2)
            elif self.current_tool == "rectangle":
                self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline=self.color, width=2)
            elif self.current_tool == "triangle":
                x1, y1 = self.start_x, self.start_y
                x2, y2 = event.x, event.y
                x3, y3 = (x1 + x2) / 2, y1 - abs(y2 - y1)
                self.canvas.create_polygon(x1, y1, x2, y2, x3, y3, outline=self.color, fill="", width=2)

        self.start_x, self.start_y = None, None

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingBoard(root)
    root.mainloop()
