"""
Tests for tools/scan_sessions.py — trigger-pulse counting from voltage CSV.

scan_sessions.py is a runnable script; the __main__ guard added in the
module lets us import count_triggers_from_csv cleanly without triggering
the filesystem scan.
"""
import os
import textwrap
import pytest

from scan_sessions import count_triggers_from_csv


def _write_csv(path, content):
    path.write_text(textwrap.dedent(content))


# ---------------------------------------------------------------------------
# Helper: generate a CSV with a TTL-like square wave on channel 1
# ---------------------------------------------------------------------------

def _square_wave_csv(tmp_path, session, n_pulses=5, high=3.3, low=0.0,
                     n_samples_per_state=10):
    """Write a voltage-recording CSV with *n_pulses* rising edges."""
    folder = tmp_path / session
    folder.mkdir()
    csv_path = folder / f'{session}_Cycle00001_VoltageRecording_001.csv'

    rows = ['Time(ms),Ch1\n']
    t = 0
    for _ in range(n_pulses):
        for _ in range(n_samples_per_state):
            rows.append(f'{t},{low}\n')
            t += 1
        for _ in range(n_samples_per_state):
            rows.append(f'{t},{high}\n')
            t += 1
    # end low
    for _ in range(n_samples_per_state):
        rows.append(f'{t},{low}\n')
        t += 1

    csv_path.write_text(''.join(rows))
    return str(folder), session


class TestCountTriggersFromCsv:
    def test_no_csv_returns_none_and_message(self, tmp_path):
        folder = tmp_path / 'empty_session'
        folder.mkdir()
        count, err = count_triggers_from_csv(str(folder), 'empty_session')
        assert count is None
        assert err is not None

    def test_correct_pulse_count(self, tmp_path):
        folder, session = _square_wave_csv(tmp_path, 'TSeries_01', n_pulses=8)
        count, err = count_triggers_from_csv(folder, session)
        assert err is None
        assert count == 8

    def test_single_pulse(self, tmp_path):
        folder, session = _square_wave_csv(tmp_path, 'TSeries_02', n_pulses=1)
        count, err = count_triggers_from_csv(folder, session)
        assert err is None
        assert count == 1

    def test_no_pulses_detected_at_high_threshold(self, tmp_path):
        """All voltages below 2.5V — falls back to 0.5V threshold."""
        folder = tmp_path / 'low_rig'
        folder.mkdir()
        # square wave between 0 and 0.8V (below 2.5V but above 0.5V)
        csv_path = folder / 'low_rig_Cycle00001_VoltageRecording_001.csv'
        rows = ['Time(ms),Ch1\n']
        for i in range(40):
            v = 0.8 if (i // 5) % 2 == 1 else 0.0
            rows.append(f'{i},{v}\n')
        csv_path.write_text(''.join(rows))
        count, err = count_triggers_from_csv(str(folder), 'low_rig')
        assert err is None
        assert count > 0

    def test_flat_signal_gives_no_pulses(self, tmp_path):
        folder = tmp_path / 'flat'
        folder.mkdir()
        csv_path = folder / 'flat_Cycle00001_VoltageRecording_001.csv'
        rows = ['Time(ms),Ch1\n'] + [f'{i},0.0\n' for i in range(100)]
        csv_path.write_text(''.join(rows))
        count, err = count_triggers_from_csv(str(folder), 'flat')
        assert count is None
        assert 'no pulses' in err

    def test_malformed_csv_returns_error(self, tmp_path):
        folder = tmp_path / 'bad'
        folder.mkdir()
        csv_path = folder / 'bad_Cycle00001_VoltageRecording_001.csv'
        csv_path.write_text('not,valid,csv\ndata\n')
        count, err = count_triggers_from_csv(str(folder), 'bad')
        # Either parse fails or no pulses found — either way not None error
        assert count is None or err is not None

    def test_picks_channel_with_most_pulses(self, tmp_path):
        """When multiple channels present, the one with most pulses wins."""
        folder = tmp_path / 'multichan'
        folder.mkdir()
        csv_path = folder / 'multichan_Cycle00001_VoltageRecording_001.csv'
        rows = ['Time(ms),Ch1,Ch2\n']
        # Ch1: 2 pulses, Ch2: 5 pulses
        for i in range(100):
            ch1 = 3.3 if (i // 10) % 2 == 1 and i < 40 else 0.0   # 2 rising edges
            ch2 = 3.3 if (i // 4) % 2 == 1 else 0.0                 # ~12 rising edges
            rows.append(f'{i},{ch1},{ch2}\n')
        csv_path.write_text(''.join(rows))
        count, err = count_triggers_from_csv(str(folder), 'multichan')
        assert err is None
        # Should pick ch2 (more pulses)
        assert count > 2
