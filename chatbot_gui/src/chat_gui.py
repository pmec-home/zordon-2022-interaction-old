#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tkinter
from tkinter import *
import format_to_ros as ftr
import asyncio

class GUI():
    # TODO: make position of buttons relative to width and height
    def __init__(self, rasa_agent, title="Title", dimension: str = "400x500",
                exit_msg = "turn off the beat"):
        self.agent = rasa_agent
        self.exit_msg = exit_msg
        self.base = Tk()
        self.base.title(title)
        self.base.geometry(dimension)
        self.base.resizable(width=FALSE, height=FALSE)
        self.width = int(dimension.split("x")[0])
        self.height = int(dimension.split("x")[1])

        # create Chat window
        self.ChatLog = Text(self.base, bd=0, bg="white", height="8", width="50", font="Arial")
        self.ChatLog.config(state=DISABLED)

        # bind scrollbar to Chat window
        scrollbar = Scrollbar(self.base, command=self.ChatLog.yview, cursor="heart")
        self.ChatLog['yscrollcommand'] = scrollbar.set

        # create the box to enter message
        self.EntryBox = Text(self.base, bd=0, bg="white", width="29", height="5", font="Arial")
        self.EntryBox.focus()

        # bind enter key to send() method
        self.base.bind('<Return>', self.send)
        self.EntryBox.bind('<Button-1>', self.send)

        # create Button to send message
        SendButton = Button(self.base, font=("Verdana",12,'bold'), text="Send",
        width="12", height=5, bd=0, bg="#32de97", activebackground="#3c9d9b",
        fg='#ffffff', command = self.send)

        # place all components on the screen
        scrollbar.place(x=376, y=6, height=386)
        self.ChatLog.place(x=6, y=6, height=386, width=370)
        self.EntryBox.place(x=128, y=401, height=90, width=265)
        SendButton.place(x=6, y=401, height=90, width=117)


    def send(self, event=None):
        msg = self.EntryBox.get("1.0",'end-1c').strip()
        self.EntryBox.delete("0.0",END)

        if msg == self.exit_msg:
            self.exit()
        elif msg != '':
            self.ChatLog.config(state=NORMAL)
            self.ChatLog.insert(END, "You: " + msg + '\n\n')
            self.ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

            # get chatbot response from msg
            # nlu_result = asyncio.run(self.agent.parse_message(msg))
            # bot_answer = asyncio.run(self.agent.handle_text(msg))

            # NOTE: make sure this is how we get the message from bot answer
            # bot_msg = ftr.get_bot_answer(bot_answer)
            # self.ChatLog.insert(END, "Bot: " + bot_msg + '\n\n')

            self.ChatLog.config(state=DISABLED)
            self.ChatLog.yview(END)

            # NOTE: copied the prints so that it goes on terminal
            # print(ftr.get_intent(nlu_result))
            # print(ftr.get_entity(nlu_result))
            # #print(ftr.get_bot_answer(bot_answer))
            # print(bot_answer)


    # TODO: make sure this works lol
    def exit(self):
        self.base.destroy()


    def start(self):
        self.base.mainloop()
