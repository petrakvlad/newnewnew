# newnewnew

## I just want the code (no browser needed)

1. Look for your terminal window. It is the black box where you type words.
2. Type `ls` and press **Enter**. You will see a folder called `notebooks`.
3. Type `cd notebooks` and press **Enter** to walk into that folder.
4. Type `python - <<'PY'` and press **Enter**. (Yes, that weird line is on purpose.)
5. Now paste the text below exactly, then press **Enter** at the end:

   ```
   import json
   from pathlib import Path

   path = Path('player_behavior_analysis.ipynb')
   data = json.loads(path.read_text())

   for number, cell in enumerate(data.get('cells', []), start=1):
       if cell.get('cell_type') == 'code':
           print(f"\n=== CODE CELL {number} ===\n")
           print(''.join(cell.get('source', [])))
   ```

6. Press **Enter** one more time and then hold **Ctrl + D**. Your terminal will now print every Python cell as clean text.
7. Scroll up and copy the parts you need.

If you ever get stuck, press **Ctrl + C** to stop and start again.

## Where the notebook file lives (tiny map)

1. In the terminal, type `ls` and press **Enter**. You should see a folder named `notebooks`.
2. Type `cd notebooks` and press **Enter**.
3. Type `ls` again. The file `player_behavior_analysis.ipynb` is the notebook.

## If you later learn how to open Jupyter

1. Make sure you are in the project folder (where `notebooks` lives).
2. Type `jupyter notebook` and press **Enter**.
3. A browser window will appear by itself. Click the `notebooks/` folder, then `player_behavior_analysis.ipynb`.
4. Each grey rectangle is Python code. Click inside to copy it.

## If you prefer Visual Studio Code (VS Code)

1. Open VS Code.
2. Choose **File → Open Folder…** and pick this project folder.
3. In the left panel, click `notebooks`, then `player_behavior_analysis.ipynb`.
4. The code cells appear on the right so you can copy them.
