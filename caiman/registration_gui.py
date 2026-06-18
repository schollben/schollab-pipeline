import os
import numpy as np
import wx
import wx.lib.agw.multidirdialog as MDD

from pipeline_job import folders_missing_registered_h5


class DirectorySelection:
    def get_directories():
        app = wx.App()
        dlg = MDD.MultiDirDialog(
            None, "Pick your dirs", defaultPath="/mnt/bigdata",
            style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST
        )
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
        self.toggle_buttons = []
        self.caiman_header_labels = []
        self.init_ui()

    def init_ui(self):
        self.panel = wx.ScrolledWindow(self)
        self.panel.SetScrollRate(0, 20)

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Skip CaImAn — greys out registration columns; FAST still runs when registered.h5 exists.
        mode_panel = wx.Panel(self.panel)
        mode_sizer = wx.BoxSizer(wx.VERTICAL)
        self.skip_caiman_cb = wx.CheckBox(
            mode_panel,
            label='Skip CaImAn (FAST only)'
        )
        self.skip_caiman_cb.Bind(wx.EVT_CHECKBOX, self.on_skip_caiman_changed)
        mode_sizer.Add(self.skip_caiman_cb, flag=wx.ALL, border=5)
        mode_sizer.Add(
            wx.StaticText(
                mode_panel,
                label='Requires registered.h5 in each folder. Remove _fast_complete to re-run FAST.'
            ),
            flag=wx.LEFT | wx.RIGHT | wx.BOTTOM,
            border=5
        )
        mode_panel.SetSizer(mode_sizer)
        main_sizer.Add(mode_panel, flag=wx.EXPAND | wx.ALL, border=5)

        caiman_header = wx.StaticText(self.panel, label='CaImAn (registration)')
        self.caiman_header_labels.append(caiman_header)
        main_sizer.Add(caiman_header, flag=wx.LEFT | wx.TOP, border=10)

        grid_sizer = wx.GridBagSizer(vgap=5, hgap=10)

        grid_sizer.Add(
            wx.StaticText(self.panel, label="Path"),
            pos=(0, 0), flag=wx.ALL, border=5
        )

        self.checkboxes = [[] for _ in range(self.check_cols)]
        for i in range(self.check_cols):
            header_sizer = wx.BoxSizer(wx.VERTICAL)
            col_label = wx.StaticText(self.panel, label=self.labels[i])
            self.caiman_header_labels.append(col_label)
            header_sizer.Add(col_label, flag=wx.ALIGN_CENTER)
            toggle_btn = wx.Button(self.panel, label="Toggle all")
            toggle_btn.Bind(wx.EVT_BUTTON, lambda evt, col=i: self.on_toggle_column(evt, col))
            self.toggle_buttons.append(toggle_btn)
            header_sizer.Add(toggle_btn, flag=wx.ALIGN_CENTER | wx.TOP, border=5)
            grid_sizer.Add(header_sizer, pos=(0, i + 1), flag=wx.ALL | wx.EXPAND, border=5)

        for row, path in enumerate(self.paths, 1):
            grid_sizer.Add(
                wx.StaticText(self.panel, label=path),
                pos=(row, 0), flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=5
            )
            for col in range(self.check_cols):
                checkbox = wx.CheckBox(self.panel)
                # Default TIFs->.H5 + First Rigid for full pipeline runs.
                if col in (0, 1):
                    checkbox.SetValue(True)
                self.checkboxes[col].append(checkbox)
                grid_sizer.Add(checkbox, pos=(row, col + 1), flag=wx.ALL | wx.ALIGN_CENTER, border=5)

        main_sizer.Add(grid_sizer, 1, wx.EXPAND | wx.ALL, border=10)

        button_panel = wx.Panel(self)
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        run_btn = wx.Button(button_panel, label="Run")
        run_btn.Bind(wx.EVT_BUTTON, self.on_close)
        button_sizer.Add(run_btn, 0, wx.ALL, 10)
        button_panel.SetSizer(button_sizer)

        outer_sizer = wx.BoxSizer(wx.VERTICAL)
        outer_sizer.Add(self.panel, 1, wx.EXPAND)
        outer_sizer.Add(button_panel, 0, wx.ALIGN_CENTER)

        self.panel.SetSizer(main_sizer)
        self.SetSizer(outer_sizer)
        self.SetSize((900, 650))
        self.panel.FitInside()

    def on_skip_caiman_changed(self, event):
        skip = self.skip_caiman_cb.GetValue()
        self._set_caiman_controls_enabled(not skip)
        if skip:
            for column_checkboxes in self.checkboxes:
                for checkbox in column_checkboxes:
                    checkbox.SetValue(False)
        else:
            # Restore full-pipeline defaults when re-enabling CaImAn columns.
            for col in (0, 1):
                for checkbox in self.checkboxes[col]:
                    checkbox.SetValue(True)

    def _set_caiman_controls_enabled(self, enabled):
        for widget in self.caiman_header_labels:
            widget.Enable(enabled)
        for toggle_btn in self.toggle_buttons:
            toggle_btn.Enable(enabled)
        for column_checkboxes in self.checkboxes:
            for checkbox in column_checkboxes:
                checkbox.Enable(enabled)

    def on_toggle_column(self, event, column):
        if not self.checkboxes[column][0].IsEnabled():
            return
        should_check = not all(cb.GetValue() for cb in self.checkboxes[column])
        for checkbox in self.checkboxes[column]:
            checkbox.SetValue(should_check)

    def on_close(self, event):
        skip_caiman = self.skip_caiman_cb.GetValue()
        if skip_caiman:
            missing = folders_missing_registered_h5(self.paths)
            if missing:
                preview = '\n'.join(f'  {p}' for p in missing[:8])
                if len(missing) > 8:
                    preview += f'\n  ... and {len(missing) - 8} more'
                dlg = wx.MessageDialog(
                    self,
                    f'{len(missing)} folder(s) have no registered.h5 — FAST will skip them:\n\n{preview}\n\nContinue anyway?',
                    'Skip CaImAn (FAST only)',
                    wx.YES_NO | wx.ICON_WARNING
                )
                if dlg.ShowModal() != wx.ID_YES:
                    dlg.Destroy()
                    return
                dlg.Destroy()

        selections = []
        for column_checkboxes in self.checkboxes:
            checked_paths = [
                self.paths[i] for i, checkbox in enumerate(column_checkboxes)
                if checkbox.GetValue()
            ]
            selections.append(checked_paths)

        self.skip_caiman = skip_caiman
        self.final_selections = selections
        self.Close()


def get_registration_options():
    paths = DirectorySelection.get_directories()
    checklist_labels = ["TIFs->.H5", "First Rigid", "Addl. Rigid", "NoRMCorre"]
    if not paths:
        return None

    app = wx.App()
    frame = CheckListFrame(paths, checklist_labels)
    frame.Show()
    app.MainLoop()
    selections = getattr(frame, 'final_selections', None)
    if selections is None:
        return None

    skip_caiman = getattr(frame, 'skip_caiman', False)
    do_h5 = np.array([(path in selections[0]) for path in paths])
    do_rig_1 = np.array([(path in selections[1]) for path in paths])
    do_rig_2 = np.array([(path in selections[2]) for path in paths])
    do_nonrig = np.array([(path in selections[3]) for path in paths])
    proc_opts = np.vstack((do_h5, do_rig_1, do_rig_2, do_nonrig))
    return np.array(paths), proc_opts, skip_caiman


def get_h5_size(h5_path):
    '''
    Just return the size of the given .h5 file.

    Parameters:
        h5_path(str): Path to h5 file.
    Returns:
        dims (tuple or None): Dimensions of the file.
            If file does not exist, return None.
    '''
    import h5py
    assert h5_path.endswith(('.h5', '.hdf5')), f"{h5_path} does not end with .h5 or .hdf5."
    try:
        with h5py.File(h5_path, 'r') as f:
            key = 'mov' if 'mov' in f.keys() else 'data'
            return f[key].shape
    except FileNotFoundError:
        print(f"Cannot give size for {h5_path} because it wasn't found.")
        return None


if __name__ == '__main__':
    user_selections = get_registration_options()
    if user_selections:
        paths, proc_opts, skip_caiman = user_selections
        print(f"\nskip_caiman: {skip_caiman}")
        print("\nFinal selections:")
        for i, row in enumerate(proc_opts):
            print(f"\nChecklist {i + 1} selections:")
            for path in paths[row]:
                print(f"  {path}")
