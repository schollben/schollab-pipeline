"""
Combined pipeline run log — per-folder summary blocks appended to repo log/.

Why separate from verbose tee: operators need a skimmable digest (CaImAn + FAST
per folder) without reading journalctl or multi-GB verbose logs.
"""
import os
from datetime import datetime

CAIMAN_STEP_LABELS = ('TIFs→H5', 'First Rigid', 'Addl. Rigid', 'NoRMCorre')
_SEP = '=' * 80


def format_run_id_ts(run_id):
	"""YYYYMMDDHHMMSS → YYYYMMDD-HHMMSS for log filenames."""
	rid = str(run_id)[:14]
	if len(rid) == 14 and rid.isdigit():
		return f'{rid[:8]}-{rid[8:14]}'
	return str(run_id)


def log_paths_for_run(repo_dir, run_id):
	"""
	Paths for one pipeline run under repo log/{timestamp}/ (not FAST install dir).

	Each run gets its own subdirectory; summary, verbose, and FAST logs live together.
	"""
	ts = format_run_id_ts(run_id)
	run_dir = os.path.join(repo_dir, 'log', ts)
	os.makedirs(run_dir, exist_ok=True)
	return {
		'run_log_dir': run_dir,
		'run_log_path': os.path.join(run_dir, 'summary.log'),
		'verbose_log_path': os.path.join(run_dir, 'verbose.log'),
		'fast_log_path': os.path.join(run_dir, 'fast.log'),
	}


def attach_log_paths(job, repo_dir):
	"""Ensure job JSON has all log paths; batch_log_path aliases verbose log."""
	paths = log_paths_for_run(repo_dir, job['run_id'])
	job.update(paths)
	# Scheduled batch code still reads batch_log_path for verbose tee.
	job['batch_log_path'] = paths['verbose_log_path']
	return job


def _format_duration(seconds):
	"""Human-readable wall duration for folder blocks."""
	if seconds >= 3600:
		h = seconds / 3600
		return f'{h:.1f} h ({seconds:.0f} s)'
	if seconds >= 60:
		return f'{seconds / 60:.1f} min ({seconds:.0f} s)'
	return f'{seconds:.1f} s'


def _artifact_line(path):
	"""Basename plus GB size when path is a file."""
	if not path or not os.path.isfile(path):
		return None
	size_gb = os.path.getsize(path) / 1e9
	return f'{os.path.basename(path)} ({size_gb:.2f} GB)'


def _format_step_line(name, step, name_width=16):
	"""One indented operation line: name status duration artifacts."""
	status = step.get('status', '?')
	dur = step.get('duration_s', 0)
	parts = [f'  {name:<{name_width}} {status:<7} {dur:>7.1f} s']
	detail = step.get('detail') or step.get('artifacts_line')
	if detail:
		parts.append(f' → {detail}')
	return ''.join(parts)


def overall_outcome_line(caiman_summary, fast_summary, skip_caiman=False):
	"""Single OVERALL line from CaImAn and FAST result enums."""
	c_res = (caiman_summary or {}).get('result', 'skipped')
	f_res = (fast_summary or {}).get('result', 'not_run')

	if skip_caiman:
		c_label = 'skipped'
	elif c_res == 'succeeded':
		c_label = 'succeeded'
	elif c_res == 'failed':
		c_label = 'failed'
	elif c_res == 'incomplete':
		c_label = 'incomplete'
	else:
		c_label = 'skipped'

	if f_res == 'succeeded':
		if c_label in ('skipped', 'succeeded'):
			return f'OVERALL: CaImAn {c_label}; FAST succeeded — folder fully complete'
		return f'OVERALL: CaImAn {c_label}; FAST succeeded — folder fully complete'
	if f_res == 'skipped':
		reason = (fast_summary or {}).get('reason', 'already complete')
		if 'complete' in reason or '_fast_complete' in reason:
			return (
				f'OVERALL: CaImAn {c_label}; FAST skipped (already complete) '
				f'— folder fully complete'
			)
		return f'OVERALL: CaImAn {c_label}; FAST skipped ({reason})'
	if f_res == 'failed':
		if c_label == 'failed':
			return 'OVERALL: CaImAn failed; FAST was not run'
		return 'OVERALL: CaImAn succeeded; FAST failed — folder incomplete'
	# not_run
	if c_label == 'failed':
		return 'OVERALL: CaImAn failed; FAST was not run'
	if c_label == 'incomplete':
		return 'OVERALL: CaImAn incomplete; FAST was not run'
	if c_label == 'succeeded':
		return 'OVERALL: CaImAn succeeded; FAST was not run'
	return f'OVERALL: CaImAn {c_label}; FAST was not run'


def write_run_header(path, *, run_id, unit_name, sessions, skip_caiman, batch_id=None):
	"""Write run-level header once at worker start."""
	started = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	lines = [
		'#' * 80,
		f'Pipeline run  run_id={run_id}  started {started}',
	]
	if batch_id:
		lines.append(f'batch_id={batch_id}')
	lines.append(
		f'Folders: {len(sessions)}  |  skip_caiman={str(skip_caiman).lower()}  '
		f'|  unit={unit_name}'
	)
	for folder in sessions:
		lines.append(f'  {folder}')
	lines.append('#' * 80)
	lines.append('')
	_append_lines(path, lines)


def append_folder_block(
	path,
	*,
	folder_idx,
	n_folders,
	folder,
	wall_s,
	caiman_summary,
	fast_summary,
	skip_caiman=False,
):
	"""Append one folder digest block; flushed immediately for crash resilience."""
	finished = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	lines = [
		_SEP,
		f'Folder {folder_idx}/{n_folders}  |  finished {finished}',
		f'Path: {folder}',
		f'Wall time: {_format_duration(wall_s)}',
		'',
		'CaImAn',
	]
	lines.extend(_format_caiman_section(caiman_summary, skip_caiman))
	lines.append('')
	lines.append('FAST')
	lines.extend(_format_fast_section(fast_summary))
	lines.append('')
	lines.append(overall_outcome_line(caiman_summary, fast_summary, skip_caiman))
	lines.append(_SEP)
	lines.append('')
	_append_lines(path, lines)


def write_run_footer(path, *, run_log_path, wall_s, folder_outcomes, run_log_dir=None):
	"""Run-level footer with aggregate folder counts."""
	finished = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	ok = sum(1 for o in folder_outcomes if o == 'complete')
	caiman_fail = sum(1 for o in folder_outcomes if o == 'caiman_failed')
	fast_fail = sum(1 for o in folder_outcomes if o == 'fast_failed')
	incomplete = len(folder_outcomes) - ok - caiman_fail - fast_fail
	log_dir = run_log_dir or os.path.dirname(run_log_path)
	lines = [
		'#' * 80,
		f'Pipeline run complete  |  finished {finished}',
		f'Wall time: {_format_duration(wall_s)}  |  folders: {ok} ok, '
		f'{caiman_fail} caiman-failed, {fast_fail} fast-failed, {incomplete} other',
		f'Log dir: {log_dir}',
		f'Summary: {run_log_path}',
		'#' * 80,
		'',
	]
	_append_lines(path, lines)


def classify_folder_outcome(caiman_summary, fast_summary, skip_caiman=False):
	"""Bucket folder for footer counts."""
	line = overall_outcome_line(caiman_summary, fast_summary, skip_caiman)
	if 'fully complete' in line:
		return 'complete'
	if 'CaImAn failed' in line:
		return 'caiman_failed'
	if 'FAST failed' in line:
		return 'fast_failed'
	return 'other'


def _format_caiman_section(summary, skip_caiman):
	if skip_caiman:
		return ['  selected:  (skipped — skip_caiman=true)', '  result:      skipped']
	if not summary:
		return ['  result:      (no data)']
	selected = summary.get('selected') or []
	if selected:
		lines = [f'  selected:  {", ".join(selected)}']
	else:
		lines = ['  selected:  (none)']
	for step in summary.get('steps', []):
		lines.append(_format_step_line(step.get('name', '?'), step, name_width=14))
	if summary.get('error'):
		lines.append(f'  error:       {summary["error"]}')
	lines.append(f'  result:      {summary.get("result", "?")}')
	return lines


def _format_fast_section(summary):
	if not summary:
		return ['  result:      (no data)']
	result = summary.get('result', '?')
	if result == 'not_run':
		reason = summary.get('reason', 'registered.h5 missing')
		return [f'  result:      not run ({reason})']
	if result == 'skipped':
		reason = summary.get('reason', '')
		detail = summary.get('detail', '')
		line = f'  result:      skipped ({reason})'
		if detail:
			line += f'; {detail}'
		return [line]
	lines = [
		_format_step_line(s.get('name', '?'), s, name_width=18)
		for s in summary.get('steps', [])
	]
	if summary.get('error'):
		lines.append(f'  error:       {summary["error"]}')
	if summary.get('artifacts_note'):
		lines.append(f'  artifacts:   {summary["artifacts_note"]}')
	lines.append(f'  result:      {result}')
	return lines


def _append_lines(path, lines):
	os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
	with open(path, 'a', encoding='utf-8') as f:
		f.write('\n'.join(lines) + '\n')
		f.flush()
