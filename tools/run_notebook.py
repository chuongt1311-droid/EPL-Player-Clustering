"""Executes notebooks/EPL_Player_Clustering.ipynb in place and stores the outputs. Usage: python tools/run_notebook.py"""
import sys, time
import nbformat
from nbclient import NotebookClient
from pathlib import Path
path = Path(__file__).resolve().parent.parent / 'notebooks' / 'EPL_Player_Clustering.ipynb'
nb = nbformat.read(path, as_version=4); t0 = time.time()
def on_done(cell, cell_index, execute_reply, **kw): print(f'cell {cell_index + 1}/{len(nb.cells)} done at {time.time() - t0:.0f}s', flush=True)
client = NotebookClient(nb, timeout=7200, kernel_name='python3', resources={'metadata': {'path': str(path.parent)}}, on_cell_executed=on_done)
try:
    client.execute(); print('EXECUTED OK', f'{time.time() - t0:.0f}s', flush=True)
except Exception as e:
    print('FAILED:', type(e).__name__, str(e)[:1500], flush=True)
finally:
    nbformat.write(nb, path)
