import json
import numpy as np
import os

class messager():
    def __init__(self, save_path, types, json_name='msg.json'):
        self.type_name_list = types
        self.msg_list = []
        for i in range(len(types)):
            self.msg_list.append([])
        self.save_path = save_path
        self.json_name = json_name

    def save_json(self):
        with open(os.path.join(self.save_path,self.json_name).replace('\\','/'),'w+') as w:
            self.msg = {}
            for i,key in enumerate(self.type_name_list):
                self.msg[key]=self.msg_list[i]
            json.dump(self.msg,w,indent=4)

    def record_message(self,types_data):
        for i,item in enumerate(types_data):
            self.msg_list[i].append(item)