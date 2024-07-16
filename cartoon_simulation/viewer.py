import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox
from imitation_shared.utils import *
import threading
from data import *

class ImageDisplayApp:
    def __init__(self, master, dataset):
        self.master = master
        self.dataset = dataset
        self.index = 0
        self.range = 1
        self.paused = False
        self.next_button_pressed = False
        self.prev_button_pressed = False
        self.button_press_time = 0
        self.photo = None

        # Create the file selection dropdown menu
        self.file_var = tk.StringVar(master)
        self.file_var.set(dataset.get_file_for_index(0))  # Set the default value
        self.file_dropdown = tk.OptionMenu(master, self.file_var, *dataset.file_paths, command=self.on_file_selected)
        self.file_dropdown.config(width=30)
        self.file_dropdown.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        
        # Create the image label
        self.image_label = tk.Label(master)
        self.image_label.grid(row=1, column=0, columnspan=3)
        
        # Create the Prev, Pause, and Next buttons
        self.prev_button = tk.Button(master, text="<<", command=self.prev_image)
        self.prev_button.grid(row=2, column=0, padx=(30, 5), pady=10, sticky="e")
        self.prev_button.bind("<ButtonPress-1>", self.on_prev_down)
        self.prev_button.bind("<ButtonRelease-1>", self.on_prev_up)

        self.pause_button = tk.Button(master, text="Pause", command=self.toggle_pause, width=5)
        self.pause_button.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

        self.next_button = tk.Button(master, text=">>", command=self.next_image)
        self.next_button.grid(row=2, column=2, padx=(50, 5), pady=10, sticky="w")
        self.next_button.bind("<ButtonPress-1>", self.on_next_down)
        self.next_button.bind("<ButtonRelease-1>", self.on_next_up)
        
        # Create the delete button
        self.delete_button = tk.Button(master, text="Delete", command=self.delete_group)
        self.delete_button.grid(row=3, column=0, columnspan=3, pady=10, padx=10)
        
        # Create index entry field
        self.index_label = tk.Label(master, text="Index:")
        self.index_label.grid(row=4, column=0, padx=5, pady=5, sticky='e')
        self.index_entry = tk.Entry(master, width=5, textvariable=tk.StringVar(value=self.index))
        self.index_entry.grid(row=4, column=1, padx=5, pady=5, sticky='w')

        # Create go to index button
        self.goto_button = tk.Button(master, text="Go To Index", command=self.goto_index)
        self.goto_button.grid(row=4, column=2, padx=10, pady=10, sticky='w')
        
        # Disable the prev, next, and delete buttons
        self.prev_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.DISABLED)
        self.delete_button.config(state=tk.DISABLED)
        
        # Load the first image
        self.load_image()
    
    def on_file_selected(self, selected_file):
        # Find the index of the selected file
        idx = self.dataset.file_paths.index(selected_file)
        
        # Reset the index to the start of the selected file
        self.index = self.dataset.get_start_index(idx)
        self.load_image()
    
    def goto_index(self):
        if not self.paused:
            self.toggle_pause()
            
        try:
            new_index = int(self.index_entry.get())
            if 0 <= new_index < len(self.dataset):
                self.index = new_index
                self.load_image()
            else:
                messagebox.showwarning("Invalid Index", "Please enter a valid index.")
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid integer index.")

    def toggle_pause(self):
        self.paused = not self.paused
        
        if not self.paused:
            self.pause_button['text'] = "Pause"
            self.prev_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
            self.delete_button.config(state=tk.DISABLED)
            self.play_image()
        else:
            self.pause_button['text'] = "Play"
            self.prev_button.config(state=tk.NORMAL)
            self.next_button.config(state=tk.NORMAL)
            self.delete_button.config(state=tk.NORMAL)
    
    def delete_group(self):
        # Create a new Toplevel window for input
        delete_window = tk.Toplevel(self.master)
        delete_window.title("Delete Groups")

        # Label and entry for start index
        start_label = tk.Label(delete_window, text="Start Index:")
        start_label.grid(row=0, column=0, padx=5, pady=5)
        start_entry = tk.Entry(delete_window)
        start_entry.grid(row=0, column=1, padx=5, pady=5)

        # Label and entry for end index
        end_label = tk.Label(delete_window, text="End Index:")
        end_label.grid(row=1, column=0, padx=5, pady=5)
        end_entry = tk.Entry(delete_window)
        end_entry.grid(row=1, column=1, padx=5, pady=5)

        # Function to perform deletion
        def perform_deletion():
            try:
                start_index = int(start_entry.get())
                end_index = int(end_entry.get())
                if start_index < 0 or end_index >= len(self.dataset):
                    messagebox.showwarning("Invalid Range", "Start or end index is out of bounds.")
                else:
                    response = messagebox.askyesno("Confirmation", f"Are you sure you want to delete groups from index {start_index} to {end_index}?")
                    if response == tk.YES:
                        # Delete the groups from the dataset
                        ret = self.dataset.delete_groups(start_index, end_index)
                        if ret == -1:
                            messagebox.showerror("Error", "Invalid range.")
                        else:
                            messagebox.showinfo("Success", f"Deleted {end_index - start_index + 1} groups.")
                            # Reload the current image
                            self.load_image()
                        delete_window.destroy()  # Close the window after deletion
            except ValueError:
                messagebox.showerror("Error", "Invalid index value. Please enter integers.")

        # Button to confirm deletion
        delete_button = tk.Button(delete_window, text="Delete", command=perform_deletion)
        delete_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

    def prev_image(self):
        self.index = max(0, self.index - 1)
        self.load_image()
        
    def on_prev_down(self, event):
        self.prev_button_pressed = True
        self.master.after(500, self.image_continuous)

    def on_prev_up(self, event):
        self.prev_button_pressed = False

    def on_next_down(self, event):
        self.next_button_pressed = True
        self.master.after(500, self.image_continuous)

    def on_next_up(self, event):
        self.next_button_pressed = False

    def image_continuous(self):
        if self.prev_button_pressed:
            self.prev_image()
        elif self.next_button_pressed:
            self.next_image()
        if self.prev_button_pressed or self.next_button_pressed:
            self.master.after(50, self.image_continuous)

    def next_image(self):
        self.index = min(len(self.dataset) - 1, self.index + 1)
        self.load_image()

    def load_image(self):
        image, scalars, targets = self.dataset[self.index]

        # Convert image from CHW to HWC format for visualization
        image = image.transpose((1, 2, 0))
        image = (image * 255).astype('uint8')

        # Overlay text on the image
        file_name = self.dataset.get_file_for_index(self.index)
        file_name_text = 'File: ' + file_name
        index_text = 'Index: ' + str(self.index)
        scalars_text = 'Scalars: ' + ', '.join([f"{x:.2f}" for x in scalars])
        targets_text = 'Targets: ' + ', '.join([f"{x:.2f}" for x in targets])
        cv2.putText(image, file_name_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
        cv2.putText(image, index_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
        cv2.putText(image, scalars_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
        cv2.putText(image, targets_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))

        # Convert image to PIL format
        image = Image.fromarray(image)

        # Convert PIL image to Tkinter PhotoImage
        self.photo = ImageTk.PhotoImage(image=image)

        # Display image in the label
        self.image_label.configure(image=self.photo)

        if not self.paused:
            self.play_image()

    def play_image(self):
        if not self.paused:
            self.master.after(10, self.next_image)


print_game_letterhead("HDF5 Data Viewer")

# Initialize Tkinter window
root = tk.Tk()
root.title("HDF5 Data Viewer")

# Load the data
dataset = ImitationDataset("data/training")

if len(dataset) == 0:
    print_formatted("No data found in the training folder", RED)
    exit()

print_formatted(f"Loaded {len(dataset)} samples from the training folder", GREEN)

# Initialize the application
app = ImageDisplayApp(root, dataset)

# Run the Tkinter event loop
root.mainloop()

print_formatted("Exiting...", RED)
