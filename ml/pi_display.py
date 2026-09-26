"""Fullscreen, sentence-only touchscreen display for Raspberry Pi.

Tkinter is included with Raspberry Pi OS's Python packages and does not need a
browser or Node.js runtime.  This module deliberately owns no camera or
recognition state: it only presents completed, already-polished sentences.
"""

import tkinter as tk


class PiSentenceDisplay:
    """A high-contrast fullscreen display for one translated sentence."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.configure(background="black")
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda _event: self.close())

        self.sentence = tk.StringVar(value="")
        self.label = tk.Label(
            self.root,
            textvariable=self.sentence,
            background="black",
            foreground="white",
            font=("DejaVu Sans", 48, "bold"),
            justify="center",
            wraplength=self.root.winfo_screenwidth() - 80,
        )
        self.label.pack(expand=True, fill="both", padx=40, pady=40)
        self.root.update_idletasks()
        self.root.update()

    def update_sentence(self, sentence):
        """Show a completed, cleaned translated sentence."""
        self.sentence.set(sentence)
        self.pump_events()

    def pump_events(self):
        """Process pending touchscreen/window events without entering mainloop."""
        self.root.update_idletasks()
        self.root.update()

    def close(self):
        """Release the Tk window and its display resources."""
        self.root.destroy()
