from POLARIScore.config import EXPORT_FOLDER
import os
from datetime import datetime
from typing import List

class Logger():
    def __init__(self, level:int=0, auto_save:int=5, save_path=None):
        """
        Args:
            level(int):Lower level means to keep only criticals informations. (0: just errors, 1: warns, 2: all).
            auto_save(int): Each n messages, a new log file will be created. Where n is "auto_save" param.
        """
        self.messages: List[str] = []
        self.level: int = level
        """Lower level means to keep only criticals informations. (0: just errors, 1: warns, 2: all)"""
        self.log_file:str = None
        """Path to the log file"""
        self.auto_save:int = auto_save
        """Interval of messages needed to save logs"""
        self.save_path:str = save_path
        self.global_color = "32m"
        self.print_borders = True

        self._init_gc = self.global_color

    def reset(self):
        """Reset logger to initial state"""
        self.global_color = self._init_gc

    def print(self, message:str, color:str="0m", type:str=None, level:int=0)->str:
        """Print a message in the console with decorations.
        Args:
            message(str): Message to print
            color(str): Color of the message in the console (python color code)
            type(str or None): if not None, prefix of the message.
            level(int): level of information.
        Returns:
            string: message_printed
        """
        if self.level < level:
            return None
        if type is None:
            string = message
        else:
            string = f"(\033[{color}{type.upper()}\033[0m) {message}"
        print(string)
        self.messages.append(string)
        if self.auto_save > 0 and len(self.messages) % self.auto_save == 0:
            self.save()
        return string
    
    def border(self, message:str="", color:str=None, level:int=2)->str:
        """
        Print a border in the console
        Args:
            message(str): a word or few words to be printed inside the border.
            color(str): color of the border in python color code.
            level(int)
        Returns:
            string: border_printed
        """
        if not(self.print_borders):
            return ""
        if color is None:
            color = self.global_color
        WIDTH = 30
        message = f" {message} "
        dashes = '-' * ((WIDTH - len(message)) // 2)
        if len(dashes) * 2 + len(message) < WIDTH:
            border_line = f"{dashes}{message}{dashes}-"
        else:
            border_line = f"{dashes}{message}{dashes}"
        return self.print(f"\033[{color}{border_line}\033[0m", level=level)
    
    def warn(self, message:str)->str:
        return self.print(message, type="warn", color="33m", level=1)
    def error(self, message:str)->str:
        message = "\033[31m"+message+"\033[0m"
        return self.print(message, type="error", color="31m", level=0)
    def log(self, message:str)->str:
        return self.print(message, type="info", color=self.global_color, level=2)

    def save(self):
        """Save messages previously printed in a log file."""
        assert self.save_path is not None, "Save path is not defined in Logger."
        
        export_path = os.path.join(self.save_path,"logs")
        if not(os.path.exists(export_path)):
            os.mkdir(export_path)
        self.log_file = os.path.join(export_path, f"logs_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt")
        with open(self.log_file, "w") as file:
            for message in self.messages:
                file.write(message + "\n")