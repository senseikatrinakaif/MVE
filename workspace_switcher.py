from core import pathex
from core.interact import interact as io
import argparse
import os 
import numpy as np

from datetime import datetime

import shutil
from pathlib import Path

if __name__ == "__main__":

    class fixPathAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace-dir', required=True, action=fixPathAction, dest="workspace_dir", help="workspace dir.")
    arguments = parser.parse_args()
    
    workspaces = sorted(pathex.get_all_dir_names_startswith(arguments.workspace_dir, "workspace_"))

    def_ws_is_link = True 
    def_ws_exists = False 
    
    def_ws = os.path.join(arguments.workspace_dir, "workspace")
    if os.path.exists(def_ws):
        def_ws_exists = True
        if not os.path.islink(def_ws):
            io.log_info("Warning: None link workspace detected ")
            def_ws_is_link = False

    s = """Choose workspace: \n"""
    if def_ws_exists:
        s += f"""(0) [default] \n"""
    for i, w in enumerate(workspaces):
        s += f"""({i+1}) {w}\n"""
    io.log_info(s)
    ws = np.clip(io.input_int ("", 0,), 0 if def_ws_exists else 1, len(workspaces)-1)

    ws_already_exists = True
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y%m%d_%H-%M-%S")
    if def_ws_exists and not def_ws_is_link:
        while(ws_already_exists):
            
            def_ws_rename = io.input_str(f"Rename existing def workspace", date_time)
            ws_already_exists = def_ws_rename in workspaces
            if ws_already_exists:
                io.log_info("Already exists, try again")

            old_path = def_ws
            new_path = os.path.join(arguments.workspace_dir, "workspace_"+def_ws_rename.lower())
            if not Path(new_path).exists():
                shutil.move(def_ws, new_path)
            else:
                raise ValueError(f"{new_path} already exists ")

    selected_ws = os.path.join(arguments.workspace_dir, "workspace_"+workspaces[ws-1])
    if Path(selected_ws).exists():
        if def_ws_exists and  def_ws_is_link:
            Path(def_ws).unlink()
        try:
            os.symlink(selected_ws, def_ws)
            io.log_info(f"Sucesfully {selected_ws} symlinked as {def_ws}")
        except Exception as ex:
            io.log_err("Could not create link! Run bat as adminstrator (e.g. via link) ")
    
        


