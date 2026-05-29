"""
scan_sessions.py
----------------
Scans all recording sessions and reports:
  - Condition structure from MarkPoints XMLs (sham/opto, laser powers)
  - Trial counts from voltage recording CSVs (trigger pulse counting)
  - H5 trial counts if already processed

Set RAW_ROOT and PROCESSED_DIR before running.
"""

import os
import glob 
import xml.etree.ElementTree as ET
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

######## Parameters ########
RAW_ROOT      = '/mnt/bigdata/BRUKER'      # set to root of raw data (where TSeries- folders are found)
PROCESSED_DIR = '/mnt/bigdata/PROCESSED'   # set if H5s exist on this machine


######## Count triggers from voltage recording CSV ########
def count_triggers_from_csv(session_folder, session_name, threshold=2.5):
    """
    Finds the voltage recording CSV for cycle 1 and counts trigger pulses.
    Prairie View stores a TTL-like signal on one of the input channels.
    We detect rising edges above threshold (default 2.5V for TTL).
    Returns (n_sham, n_opto) if 2 conditions detected, else total count.
    """
    csv_pattern = os.path.join(
        session_folder, f'{session_name}_Cycle00001_VoltageRecording_001.csv'
    )
    matches = glob.glob(csv_pattern)
    if not matches:
        # Try any voltage recording CSV
        matches = glob.glob(os.path.join(session_folder, '*VoltageRecording*.csv'))
    if not matches:
        return None, 'no CSV found'

    try:
        # Read header to find channel names
        with open(matches[0], 'r') as f:
            lines = f.readlines()

        # Find header line (contains 'Time' or channel names)
        header_idx = 0
        for i, line in enumerate(lines):
            if 'Time' in line or 'time' in line or 'ms' in line.lower():
                header_idx = i
                break

        # Load numeric data
        data = np.genfromtxt(matches[0], delimiter=',',
                             skip_header=header_idx + 1)
        if data.ndim == 1 or data.shape[1] < 2:
            return None, 'CSV parse failed'

        # Try each channel (skip time column) for TTL-like pulses
        best_count = 0
        for col in range(1, min(data.shape[1], 6)):
            sig    = data[:, col]
            # Rising edge detection
            above  = (sig > threshold).astype(int)
            edges  = np.diff(above)
            n_rise = int((edges == 1).sum())
            if n_rise > best_count:
                best_count = n_rise

        if best_count == 0:
            # Try lower threshold (some rigs use 0-1V logic)
            threshold = 0.5
            for col in range(1, min(data.shape[1], 6)):
                sig    = data[:, col]
                above  = (sig > threshold).astype(int)
                edges  = np.diff(above)
                n_rise = int((edges == 1).sum())
                if n_rise > best_count:
                    best_count = n_rise

        if best_count == 0:
            return None, 'no pulses detected'

        # If 2 conditions, total triggers split evenly
        return best_count, None

    except Exception as e:
        return None, f'error: {e}'


######## Find all sessions ########
all_mp = sorted(glob.glob(
    os.path.join(RAW_ROOT, '**', 'TSeries-*_Cycle00001_MarkPoints.xml'),
    recursive=True
))

if not all_mp:
    raise FileNotFoundError(f'No MarkPoints XMLs found under {RAW_ROOT}')

sessions = {}
for path in all_mp:
    fname   = os.path.basename(path)
    session = fname.replace('_Cycle00001_MarkPoints.xml', '')
    sessions[session] = os.path.dirname(path)

print(f'{"="*90}')
print(f'{"SESSION":<45} {"COND":<6} {"POWERS":<14} '
      f'{"TRIGGERS":<12} {"TRIALS(ea)":<12} {"FLAGS"}')
print(f'{"-"*90}')

for session, folder in sorted(sessions.items()):
    flags = []

    # --- MarkPoints XML ---
    mp_path = os.path.join(folder, f'{session}_Cycle00001_MarkPoints.xml')
    try:
        tree     = ET.parse(mp_path)
        root     = tree.getroot()
        elements = root.findall('PVMarkPointElement')
        n_cond   = len(elements)
        powers   = [float(el.attrib.get('UncagingLaserPower', -1))
                    for el in elements]
        has_sham = 0.0 in powers
        has_opto = any(p > 0 for p in powers)

        if not has_sham: flags.append('NO SHAM')
        if not has_opto: flags.append('NO OPTO')
        if n_cond > 2:   flags.append(f'{n_cond} CONDITIONS')

        cond_str  = str(n_cond)
        power_str = '/'.join(str(int(p)) for p in powers)
    except Exception as e:
        cond_str = power_str = 'ERR'
        flags.append(f'XML error')

    # --- Voltage recording CSV for trigger count ---
    n_triggers, csv_err = count_triggers_from_csv(folder, session)
    if csv_err:
        trig_str  = '?'
        trial_str = f'({csv_err})'
    else:
        trig_str = str(n_triggers)
        if n_cond == 2 and has_sham and has_opto:
            n_each    = n_triggers // 2
            trial_str = f'~{n_each} each'
        else:
            trial_str = f'total={n_triggers}'

    # --- H5 cross-check if available ---
    if HAS_H5PY:
        h5_matches = glob.glob(os.path.join(PROCESSED_DIR, f'{session}.h5'))
        if h5_matches:
            try:
                with h5py.File(h5_matches[0], 'r') as f:
                    cyc   = f['cyc_photostim_only'][()]
                    mp_lp = f['markpoints_laser_power'][()] \
                            if 'markpoints_laser_power' in f else np.array([])
                    if cyc.shape[1] > 1:
                        trial_str = f'{cyc.shape[2]} ea [H5]'
                    elif len(mp_lp) > 0 and 0. in mp_lp and np.any(mp_lp > 0):
                        trial_str = f'{cyc.shape[2]//2} ea [H5-recovered]'
                        flags.append('CONDITIONS COLLAPSED')
                    else:
                        trial_str = f'total={cyc.shape[2]} [H5]'
            except Exception:
                pass

    flag_str = ' | '.join(flags) if flags else 'OK'
    print(f'{session:<45} {cond_str:<6} {power_str:<14} '
          f'{trig_str:<12} {trial_str:<12} {flag_str}')

print(f'{"="*90}')
print('Done.')
