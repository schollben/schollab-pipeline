from os.path import basename, dirname,join
import code
import numpy as np
import wx
import wx.lib.agw.multidirdialog as MDD

class DirectorySelection:
    def get_directories():
        app = wx.App() 
        dlg = MDD.MultiDirDialog(None, "Pick your dirs", defaultPath="/mnt/bigdata", style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
        if dlg.ShowModal() == wx.ID_OK:
            paths = dlg.GetPaths()
        dlg.Destroy()
        return paths if paths else None
 
class CheckListFrame(wx.Frame):
    def __init__(self, paths, checklist_labels):
        super().__init__(parent=None, title='Directory Checklist Tool')
        self.paths = paths
        self.labels = checklist_labels
        self.check_cols = len(checklist_labels)
        self.init_ui()
        
    def init_ui(self):
        # Create the main panel as a scrolled window
        self.panel = wx.ScrolledWindow(self)
        self.panel.SetScrollRate(0, 20)
        
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Create grid sizer for paths and checkboxes
        grid_sizer = wx.GridBagSizer(vgap=5, hgap=10)
        
        # Add column headers
        grid_sizer.Add(wx.StaticText(self.panel, label="Path"), 
                      pos=(0, 0), flag=wx.ALL, border=5)
        
        self.checkboxes = [[] for _ in range(self.check_cols)]
        # Add column headers and toggle buttons
        for i in range(self.check_cols):
            header_sizer = wx.BoxSizer(wx.VERTICAL)
            header_sizer.Add(wx.StaticText(self.panel, label=self.labels[i]), 
                           flag=wx.ALIGN_CENTER)
            toggle_btn = wx.Button(self.panel, label=f"Toggle all")
            toggle_btn.Bind(wx.EVT_BUTTON, lambda evt, col=i: self.on_toggle_column(evt, col))
            header_sizer.Add(toggle_btn, flag=wx.ALIGN_CENTER | wx.TOP, border=5)
            grid_sizer.Add(header_sizer, pos=(0, i+1), 
                          flag=wx.ALL | wx.EXPAND, border=5)

        # Add paths and checkboxes
        for row, path in enumerate(self.paths, 1):
            # Add path
            grid_sizer.Add(wx.StaticText(self.panel, label=path), 
                          pos=(row, 0), flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5)
            
            # Add checkboxes for each column
            for col in range(self.check_cols):
                checkbox = wx.CheckBox(self.panel)
                # Default TIFs->.H5 + First Rigid so motion correction runs and produces
                # registered.h5 for FAST when the user accepts defaults without extra clicks.
                if col in (0, 1):
                    checkbox.SetValue(True)
                self.checkboxes[col].append(checkbox)
                grid_sizer.Add(checkbox, pos=(row, col+1), 
                             flag=wx.ALL | wx.ALIGN_CENTER, border=5)

        main_sizer.Add(grid_sizer, 1, wx.EXPAND | wx.ALL, 10)
        
        # Bottom button panel (not scrolled)
        # "Run" captures selections and closes — actual pipeline launch happens
        # in registration.py after get_registration_options() returns
        button_panel = wx.Panel(self)
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        run_btn = wx.Button(button_panel, label="Run")
        run_btn.Bind(wx.EVT_BUTTON, self.on_close)
        button_sizer.Add(run_btn, 0, wx.ALL, 10)
        button_panel.SetSizer(button_sizer)
        
        # Outer sizer to combine scrolled content and button
        outer_sizer = wx.BoxSizer(wx.VERTICAL)
        outer_sizer.Add(self.panel, 1, wx.EXPAND)
        outer_sizer.Add(button_panel, 0, wx.ALIGN_CENTER)
        
        # Set sizers
        self.panel.SetSizer(main_sizer)
        self.SetSizer(outer_sizer)
        
        # Set frame size and ensure scrollbars appear when needed
        self.SetSize((900, 600))
        self.panel.FitInside()
    
    def on_toggle_column(self, event, column):
        should_check = not all(cb.GetValue() for cb in self.checkboxes[column])
        for checkbox in self.checkboxes[column]:
            checkbox.SetValue(should_check)
    
    
    def on_close(self, event):
        selections = []
        for column_checkboxes in self.checkboxes:
            checked_paths = [
                self.paths[i] for i, checkbox in enumerate(column_checkboxes)
                if checkbox.GetValue()
            ]
            selections.append(checked_paths)
        
        self.final_selections = selections
        self.Close()
    


def get_registration_options():

    paths = DirectorySelection.get_directories()
    checklist_labels = ["TIFs->.H5", "First Rigid","Addl. Rigid", "NoRMCorre"]
    if not paths:
        return None
        
    app = wx.App()
    frame = CheckListFrame(paths, checklist_labels)
    frame.Show()
    app.MainLoop()
    selections = getattr(frame, 'final_selections', None)
    # Need to convert over to indices for ease
    do_h5     = np.array([(path in selections[0]) for path in paths])
    do_rig_1  = np.array([(path in selections[1]) for path in paths])
    do_rig_2  = np.array([(path in selections[2]) for path in paths])
    do_nonrig = np.array([(path in selections[3]) for path in paths])
    return (np.array(paths), np.vstack((do_h5, do_rig_1, do_rig_2, do_nonrig)))



def get_h5_size(h5_path):
    '''
    Just return the size of the given .h5 file.
    
    Parameters:
        h5_path(str): Path to h5 file.
    Returns:
        dims (tuple or None): Dimensions of the file.
            If file does not exist, return None.
    '''
    assert h5_path.endswith(('.h5', '.hdf5')), f"{h5_path} does not end with .h5 or .hdf5."
    try:
        with h5py.File(h5_path, 'r') as f:
            key = 'mov' if 'mov' in f.keys() else 'data'
            return f[key].shape
    except FileNotFoundError:
        print("Cannot give size for f{h5_path} because it wasn't found.")
        return None
    


# Brief test
if __name__ == '__main__':
    user_selections = get_registration_options()
    if user_selections:
        print("\nFinal selections:")
        for i, selection in enumerate(user_selections):
            print(f"\nChecklist {i+1} selections:")
            for path in selection:
                print(f"  {path}")