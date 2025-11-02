# Player Analysis Project — Plain-Language Guide

This guide explains, in very simple words, what the project files do and what you can expect to see when you run them.

## 1. The Notebook (notebooks/player_behavior_analysis.ipynb)
- Think of the notebook as a long story told in blocks.
- First, it cleans the September data so every player has a full 30-day calendar.
- Next, it builds easy-to-read numbers: how recently someone played, how often, how much money they spent, and how long they stayed away.
- It also makes rolling totals (like a 7-day moving average) so you can spot rising or falling spend.
- Every big step has a picture: heatmaps for player movement between sectors, line charts for spend over time, scatter plots for weird spikes, and bar charts that compare acquisition channels.

## 2. Helper Code (player_analysis/analytics.py)
- This Python file is a toolbox that the notebook uses to stay tidy.
- The `modeling_frame` function reshapes the data so every row knows “what happened today” and “what happened next.”
- There are training helpers for three questions:
  1. **Which sector will a player use tomorrow?** (`train_next_sector_classifier`)
  2. **How much will they spend tomorrow?** (`train_next_spend_regressor`)
  3. **Will they jump from virtual to TV or other?** (`train_cross_sector_activation`)
- Other helpers make ready-to-plot tables, such as confusion matrices, feature importance charts, residual checks, and anomaly scores.
- Sequence functions rebuild each player’s timeline so we can count popular paths like “virtual → pause → tv.”
- Channel helpers stack up statistics for every acquisition channel so you can see which one brings loyal or high-spending users.

## 3. Automated Checks (tests/test_steps_8_10.py)
- The tests create a tiny pretend dataset that looks like the real thing.
- They confirm the toolbox functions return useful numbers (for example, fold metrics, feature importance tables, residual diagnostics).
- They also check that we can rebuild sequences, flag odd spend days, and compute channel summaries without crashing.
- Running `pytest` is a quick “does everything still work?” button.

## 4. How the Pieces Fit Together
1. Load the notebook.
2. Run each cell from top to bottom.
3. When the notebook calls a helper (for example, `train_next_sector_classifier`), the helper comes from `player_analysis/analytics.py`.
4. The helper returns both the model answers and the data needed for visuals, so the notebook can draw charts right away.
5. If you ever change the helpers, run `pytest` to be sure nothing broke.

That’s it—the notebook tells the story, the helper file does the heavy lifting, and the tests make sure the story still makes sense.
