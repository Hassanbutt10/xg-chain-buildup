"""
xG Chain and Buildup Leaderboard - Cycle 8, Day 3 project
Goals and assists credit only the last two touches of a move. This project
shares the credit across everyone involved in a possession that ends in a
shot, using two standard measures:

  xG chain:    every player who passed, carried, dribbled or shot in the
               possession gets the full xG of the possession's shots
  xG buildup:  the same, but the shooter and the player who made the final
               (shot-creating) pass get nothing, so it shows who started
               and moved the play

Rules used here:
  - Copa America 2024, periods 1 to 4 (extra time included, shootouts out)
  - Penalties are excluded from both the xG measures and goals + assists
  - A possession's value is the summed xG of its shots (rebounds included)
  - A player is credited once per possession, however many touches he had

Limitations: totals, not per-90, because lineup data was not pulled, so
teams that played more matches have more chances to collect credit. Credit
adds up to more than the tournament's total xG, because one chance credits
many players. Defensive actions that win the ball are not credited unless
the player then passes, carries or shoots. The xG itself comes from
StatsBomb's shot model and says nothing about the quality of earlier passes.

Outputs: xg_chain_leaderboard.png, buildup_vs_goals_assists.png,
goal_move_credit.png, goal_move_replay.mp4 (replay plus the two charts as
end cards), xg_chain_buildup.csv

Reuses: copa_america_2024_matches.csv, copa_america_2024_events.csv
"""

import os
import re
import subprocess
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, Circle, Patch
import imageio_ffmpeg

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

FOCUS_TEAM = "Argentina"
TOP_N = 12
END_CARDS = ["xg_chain_leaderboard.png", "buildup_vs_goals_assists.png"]
END_CARD_SECONDS = 4

matches = pd.read_csv("copa_america_2024_matches.csv")
events = pd.read_csv("copa_america_2024_events.csv", low_memory=False)
events = events[events["period"].isin([1, 2, 3, 4])].copy()
for col in ["shot_type", "pass_shot_assist", "pass_goal_assist", "shot_outcome",
            "shot_end_location", "carry_end_location", "pass_end_location", "index", "minute"]:
    if col not in events.columns:
        events[col] = np.nan


def truthy(s):
    return s.astype(str).str.lower().eq("true")


# ---------- names ----------
OVERRIDES = {
    "Lionel Andrés Messi Cuccittini": "Messi",
    "Ángel Fabián Di María Hernández": "Di María",
    "Nahuel Molina Lucero": "Molina",
    "Damián Emiliano Martínez": "E. Martínez",
    "Lautaro Javier Martínez": "Lautaro Martínez",
    "Lisandro Martínez": "Lisandro Martínez",
    "Raphael Dias Belloli": "Raphinha",
    "Luis Fernando Díaz Marulanda": "Luis Díaz",
    "Marcos Aoás Corrêa": "Marquinhos",
    "Marcos Aoas Correa": "Marquinhos",
    "Éderson José dos Santos Lourenço da Silva": "Ederson",
}
PARTICLES = {"de", "da", "di", "dos", "van", "von", "mac", "lo", "la", "del", "le"}
TEAM_CODES = {
    "Argentina": "ARG", "Brazil": "BRA", "Colombia": "COL", "Uruguay": "URU", "Canada": "CAN",
    "Venezuela": "VEN", "Ecuador": "ECU", "Panama": "PAN", "Mexico": "MEX", "United States": "USA",
    "Chile": "CHI", "Peru": "PER", "Paraguay": "PAR", "Bolivia": "BOL", "Jamaica": "JAM",
    "Costa Rica": "CRC",
}


def short_name(full):
    """Common name from a long legal name. Handles particles such as De La Cruz
    and Mac Allister, and the two Brazilian 'da Silva' names."""
    if full in OVERRIDES:
        return OVERRIDES[full]
    parts = full.split()
    # first particle that is not the first name, plus the particles that follow it
    for i in range(1, len(parts) - 1):
        if parts[i].lower() in PARTICLES:
            j = i
            while j < len(parts) - 1 and parts[j].lower() in PARTICLES:
                j += 1
            surname = " ".join(parts[i:j + 1])
            if surname.lower() == "da silva":
                # Danilo Luiz da Silva -> Danilo, Joao Victor Gomes da Silva -> Joao Gomes
                return parts[0] if i == 2 else f"{parts[0]} {parts[i - 1]}"
            return surname
    if len(parts) >= 4:
        return parts[-2]
    return parts[-1]


def display_names(full_names):
    """Short names for a list of players. Two different players with the same
    short name get their first name added, unless it is already in the name."""
    uniq = list(dict.fromkeys(full_names))
    base = {p: short_name(p) for p in uniq}
    groups = {}
    for p, b in base.items():
        groups.setdefault(b, []).append(p)
    out = {}
    for p, b in base.items():
        first = p.split()[0]
        out[p] = f"{first} {b}" if len(groups[b]) > 1 and not b.startswith(first) else b
    return out


# ---------- possessions that end in a shot ----------
shots = events[(events["type"] == "Shot") & events["shot_statsbomb_xg"].notna()].copy()
shots = shots[shots["shot_type"].fillna("") != "Penalty"]
poss_xg = (shots.groupby(["match_id", "possession", "team"], as_index=False)["shot_statsbomb_xg"]
           .sum().rename(columns={"shot_statsbomb_xg": "xg", "team": "shot_team"}))
shots_per_poss = shots.groupby(["match_id", "possession"]).size()

TOUCH_TYPES = ["Pass", "Carry", "Dribble", "Shot"]
inv = events[events["type"].isin(TOUCH_TYPES) & events["player"].notna()].merge(
    poss_xg, on=["match_id", "possession"], how="inner")
inv = inv[inv["team"] == inv["shot_team"]].copy()
inv["is_shooter"] = inv["type"] == "Shot"
inv["is_key"] = (inv["type"] == "Pass") & (truthy(inv["pass_shot_assist"]) | truthy(inv["pass_goal_assist"]))

pp = inv.groupby(["match_id", "possession", "player", "team"], as_index=False).agg(
    xg=("xg", "first"), shooter=("is_shooter", "max"), key=("is_key", "max"))
pp["buildup_xg"] = np.where(pp["shooter"] | pp["key"], 0.0, pp["xg"])

board = pp.groupby(["player", "team"], as_index=False).agg(
    xg_chain=("xg", "sum"), xg_buildup=("buildup_xg", "sum"), moves=("xg", "count"))

goals = shots[shots["shot_outcome"] == "Goal"].groupby("player").size()
assists = events[(events["type"] == "Pass") & truthy(events["pass_goal_assist"])].groupby("player").size()
board["goals"] = board["player"].map(goals).fillna(0).astype(int)
board["assists"] = board["player"].map(assists).fillna(0).astype(int)
board["g_plus_a"] = board["goals"] + board["assists"]

board["name"] = board["player"].map(display_names(board["player"].tolist()))
board["label"] = board["name"] + " (" + board["team"].map(lambda t: TEAM_CODES.get(t, t[:3].upper())) + ")"
board = board.sort_values("xg_chain", ascending=False).reset_index(drop=True)
board.to_csv("xg_chain_buildup.csv", index=False)

cols = ["label", "xg_chain", "xg_buildup", "moves", "goals", "assists"]
print(f"\nPossessions ending in a non-penalty shot: {len(poss_xg)}")
print("\nTop by xG chain:")
print(board[cols].head(TOP_N).round(2).to_string(index=False))
print("\nTop by xG buildup:")
print(board.sort_values("xg_buildup", ascending=False)[cols].head(TOP_N).round(2).to_string(index=False))

BG = "#0d1b2a"
GRID = "#3a4a5a"
BLUE = "#4a90d9"
ORANGE = "#f4a300"
MUTED = "#a8b8c8"

# ---------- chart 1: chain vs buildup leaderboard ----------
top = board.head(TOP_N).iloc[::-1].reset_index(drop=True)
fig1, ax1 = plt.subplots(figsize=(11, 8))
fig1.patch.set_facecolor(BG)
ax1.set_facecolor(BG)
yy = np.arange(len(top))
ax1.barh(yy + 0.2, top["xg_chain"], height=0.38, color=BLUE, label="xG chain (everyone involved)")
ax1.barh(yy - 0.2, top["xg_buildup"], height=0.38, color=ORANGE, label="xG buildup (without shooter and final passer)")
xmax = top["xg_chain"].max()
for i, r in top.iterrows():
    ax1.text(xmax * 1.02, i, f"G+A: {int(r['g_plus_a'])}", va="center", color="white", fontsize=11)
ax1.set_yticks(yy)
ax1.set_yticklabels(top["label"], color="white", fontsize=12)
ax1.set_xlim(0, xmax * 1.16)
ax1.set_xlabel("Total xG credited over the tournament", color="white", fontsize=11)
ax1.tick_params(axis="x", colors="white")
ax1.tick_params(axis="y", length=0)
for sp in ax1.spines.values():
    sp.set_color(GRID)
ax1.legend(facecolor=BG, labelcolor="white", edgecolor=GRID, loc="upper center",
           bbox_to_anchor=(0.5, -0.1), ncol=2)
fig1.suptitle("Who Was Involved in the Most Dangerous Moves?", color="white", fontsize=17,
              fontweight="bold", y=0.97)
ax1.set_title("xG chain and buildup, Copa America 2024, penalties excluded. G+A = goals plus assists",
              color=MUTED, fontsize=10.5, pad=10)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig1.savefig("xg_chain_leaderboard.png", dpi=150, facecolor=BG)
plt.close(fig1)
print("\nSaved chart: xg_chain_leaderboard.png")

# ---------- chart 2: buildup leaders vs goals and assists ----------
tb = board.sort_values("xg_buildup", ascending=False).head(TOP_N).iloc[::-1].reset_index(drop=True)
fig2, ax2 = plt.subplots(figsize=(11, 8))
fig2.patch.set_facecolor(BG)
ax2.set_facecolor(BG)
yy = np.arange(len(tb))
ax2.barh(yy, tb["xg_buildup"], height=0.6,
         color=[ORANGE if g == 0 else BLUE for g in tb["g_plus_a"]])
bmax = tb["xg_buildup"].max()
for i, r in tb.iterrows():
    ax2.text(r["xg_buildup"] + bmax * 0.015, i, f"{r['xg_buildup']:.2f}   G+A: {int(r['g_plus_a'])}",
             va="center", color="white", fontsize=11)
ax2.set_yticks(yy)
ax2.set_yticklabels(tb["label"], color="white", fontsize=12)
ax2.set_xlim(0, bmax * 1.3)
ax2.set_xlabel("Total xG buildup over the tournament", color="white", fontsize=11)
ax2.tick_params(axis="x", colors="white")
ax2.tick_params(axis="y", length=0)
for sp in ax2.spines.values():
    sp.set_color(GRID)
ax2.legend(handles=[Patch(color=ORANGE, label="No goal or assist at the tournament"),
                    Patch(color=BLUE, label="Scored or assisted")],
           facecolor=BG, labelcolor="white", edgecolor=GRID, loc="upper center",
           bbox_to_anchor=(0.5, -0.1), ncol=2)
fig2.suptitle("The Work Behind the Chances: xG Buildup Leaders", color="white", fontsize=17,
              fontweight="bold", y=0.97)
ax2.set_title("Credit for starting and moving the play, shown next to goals plus assists (G+A)",
              color=MUTED, fontsize=10.5, pad=10)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig2.savefig("buildup_vs_goals_assists.png", dpi=150, facecolor=BG)
plt.close(fig2)
print("Saved chart: buildup_vs_goals_assists.png")

# ---------- pick one real goal move to replay ----------
opp_of = {}
for r in matches.itertuples():
    opp_of[(r.match_id, r.home_team)] = r.away_team
    opp_of[(r.match_id, r.away_team)] = r.home_team

cand = shots[(shots["team"] == FOCUS_TEAM) & (shots["shot_outcome"] == "Goal")]
best = None
for r in cand.itertuples():
    if shots_per_poss.get((r.match_id, r.possession), 0) != 1:
        continue
    sub = inv[(inv["match_id"] == r.match_id) & (inv["possession"] == r.possession)]
    score = (sub["player"].nunique(), len(sub))
    if best is None or score > best[0]:
        best = (score, r.match_id, r.possession)
if best is None:
    raise ValueError(f"No single-shot {FOCUS_TEAM} goal found to replay")
_, g_match, g_poss = best

move = events[(events["match_id"] == g_match) & (events["possession"] == g_poss)
              & (events["team"] == FOCUS_TEAM) & events["type"].isin(TOUCH_TYPES)
              & events["player"].notna()].copy()
move = move.sort_values("index") if move["index"].notna().any() else move
XY = re.compile(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)")


def parse_xy(val):
    m = XY.match(str(val))
    return (float(m.group(1)), float(m.group(2))) if m else None


actions = []
for r in move.itertuples():
    start = parse_xy(r.location)
    if start is None:
        continue
    if r.type == "Pass":
        end = parse_xy(r.pass_end_location)
    elif r.type == "Carry":
        end = parse_xy(r.carry_end_location)
    elif r.type == "Shot":
        end = parse_xy(r.shot_end_location)
    else:
        end = start
    if end is None:
        end = start
    actions.append(dict(player=r.player, type=r.type, start=start, end=end))
    if r.type == "Shot":
        break

shot_row = move[move["type"] == "Shot"].iloc[0]
goal_xg = float(poss_xg[(poss_xg["match_id"] == g_match) & (poss_xg["possession"] == g_poss)]["xg"].iloc[0])
goal_minute = int(shot_row["minute"]) if pd.notna(shot_row["minute"]) else 0
opponent = opp_of.get((g_match, FOCUS_TEAM), "")

# who is involved, in order of first touch, and what each earns
order_players, first_pos = [], {}
for a in actions:
    if a["player"] not in first_pos:
        first_pos[a["player"]] = a["start"]
        order_players.append(a["player"])
pp_goal = pp[(pp["match_id"] == g_match) & (pp["possession"] == g_poss)].set_index("player")
disp = display_names(order_players)
rows = []
for pl in order_players:
    rw = pp_goal.loc[pl]
    role = "shot" if rw["shooter"] else ("key pass" if rw["key"] else "buildup")
    rows.append((disp[pl], role, goal_xg, None if role != "buildup" else goal_xg))

print(f"\nGoal move replayed: {FOCUS_TEAM} vs {opponent}, minute {goal_minute}, "
      f"{len(order_players)} players, {len(actions)} actions, xG {goal_xg:.2f}")
for r in rows:
    print(f"  {r[0]:<18}{r[1]:<10}chain {r[2]:.2f}  buildup {'-' if r[3] is None else f'{r[3]:.2f}'}")


# ---------- drawing the replay ----------
def draw_pitch(ax):
    ax.set_facecolor("#10253a")
    kw = dict(fill=False, edgecolor="#35506b", linewidth=1.6)
    ax.add_patch(Rectangle((0, 0), 120, 80, **kw))
    ax.plot([60, 60], [0, 80], color="#35506b", linewidth=1.6)
    ax.add_patch(Circle((60, 40), 10, **kw))
    ax.add_patch(Rectangle((0, 18), 18, 44, **kw))
    ax.add_patch(Rectangle((102, 18), 18, 44, **kw))
    ax.add_patch(Rectangle((0, 30), 6, 20, **kw))
    ax.add_patch(Rectangle((114, 30), 6, 20, **kw))
    ax.set_xlim(-2, 122)
    ax.set_ylim(82, -2)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def place_labels(xs, ys, labels, ppu=6.7, radius_units=2.2):
    """Choose a label spot next to each dot that avoids other labels and dots."""
    placed, result = [], {}
    for i in sorted(range(len(xs)), key=lambda k: -sum(
            np.hypot(xs[k] - xs[j], ys[k] - ys[j]) < 14 for j in range(len(xs)) if j != k)):
        w = (len(labels[i]) * 10.6 + 12) / ppu
        h = 24 / ppu
        r = radius_units
        near = [(0, -(r + h / 2 + 0.8)), (0, r + h / 2 + 0.8),
                (r + w / 2 + 0.8, 0), (-(r + w / 2 + 0.8), 0),
                (r * 0.7 + w * 0.3, -(r * 0.7 + h / 2)), (-(r * 0.7 + w * 0.3), -(r * 0.7 + h / 2)),
                (r * 0.7 + w * 0.3, r * 0.7 + h / 2), (-(r * 0.7 + w * 0.3), r * 0.7 + h / 2)]
        cands = [(dx, dy, 1.0) for dx, dy in near] + [(dx * 1.9, dy * 1.9, 2.0) for dx, dy in near]
        best_c = None
        for ci, (dx, dy, sc) in enumerate(cands):
            cx, cy = xs[i] + dx, ys[i] + dy
            pen = (ci % 8) * 0.4 + (sc - 1.0) * 4.0
            if cx - w / 2 < -2 or cx + w / 2 > 122 or cy - h / 2 < -2 or cy + h / 2 > 82:
                pen += 50
            for (ox, oy, ow, oh) in placed:
                pen += 12 * max(0.0, (w + ow) / 2 - abs(cx - ox)) * max(0.0, (h + oh) / 2 - abs(cy - oy))
            for j in range(len(xs)):
                pen += 6 * max(0.0, w / 2 + r - abs(cx - xs[j])) * max(0.0, h / 2 + r - abs(cy - ys[j]))
            if best_c is None or pen < best_c[0]:
                best_c = (pen, cx, cy)
        _, cx, cy = best_c
        gap = np.hypot(max(abs(cx - xs[i]) - w / 2, 0.0), max(abs(cy - ys[i]) - h / 2, 0.0))
        result[i] = (cx, cy, gap > r + 1.5)
        placed.append((cx, cy, w, h))
    return result


fig = plt.figure(figsize=(10, 10), dpi=108)  # 1080 x 1080 pixels
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0.03, 0.36, 0.94, 0.52])

fig.text(0.5, 0.955, f"Who Gets Credit for This {FOCUS_TEAM} Goal?", ha="center",
         color="white", fontsize=25, fontweight="bold")
fig.text(0.5, 0.915, f"{FOCUS_TEAM} vs {opponent}, minute {goal_minute}", ha="center",
         color=MUTED, fontsize=15)
line1_t = fig.text(0.5, 0.29, "", ha="center", color=ORANGE, fontsize=21, fontweight="bold")
chain_t1 = fig.text(0.5, 0.215, "", ha="center", color="white", fontsize=16)
chain_t2 = fig.text(0.5, 0.165, "", ha="center", color="white", fontsize=16)
caption_t = fig.text(0.5, 0.335, "xG chain: everyone involved gets the xG of the shot. "
                     "xG buildup: the same, minus the shooter and the final passer.",
                     ha="center", color=MUTED, fontsize=10.5)
fig.text(0.5, 0.012, "StatsBomb open data, Copa America 2024. xG from StatsBomb's shot model.",
         ha="center", color=MUTED, fontsize=9)

MONO = "DejaVu Sans Mono"
n_rows = len(rows)
step = min(0.0255, 0.235 / (n_rows + 1))
table_texts = []
header = f"{'Player':<19}{'Role':<10}{'xG chain':>9}{'xG buildup':>12}"
table_texts.append(fig.text(0.5, 0.305, header, ha="center", va="center", family=MONO,
                            color=MUTED, fontsize=13, fontweight="bold"))
for i, (nm, ro, ch, bu) in enumerate(rows):
    txt = f"{nm[:18]:<19}{ro:<10}{ch:>9.2f}{('-' if bu is None else f'{bu:.2f}'):>12}"
    color = ORANGE if ro == "shot" else ("#6fcf97" if ro == "buildup" else "white")
    table_texts.append(fig.text(0.5, 0.305 - step * (i + 1), txt, ha="center", va="center",
                                family=MONO, color=color, fontsize=13))

xs_p = np.array([first_pos[pl][0] for pl in order_players])
ys_p = np.array([first_pos[pl][1] for pl in order_players])
label_pos = place_labels(xs_p, ys_p, [disp[pl] for pl in order_players])
player_idx = {pl: i for i, pl in enumerate(order_players)}
first_action = {pl: next(k for k, a in enumerate(actions) if a["player"] == pl) for pl in order_players}


def draw_frame(done, cur, t, line1, rows_vis):
    ax.clear()
    draw_pitch(ax)
    last_vis = cur if cur is not None else len(actions) - 1
    for j in range(done):
        a = actions[j]
        style = dict(color="white", lw=2.6, ls="-") if a["type"] == "Pass" else (
            dict(color="#9db4c9", lw=2.6, ls="--") if a["type"] == "Carry" else
            dict(color=ORANGE, lw=3.6, ls="-"))
        if a["type"] != "Dribble":
            ax.plot([a["start"][0], a["end"][0]], [a["start"][1], a["end"][1]],
                    color=style["color"], linewidth=style["lw"], linestyle=style["ls"],
                    alpha=0.9, zorder=2)
    ball = None
    if cur is not None:
        a = actions[cur]
        bx = a["start"][0] + (a["end"][0] - a["start"][0]) * t
        by = a["start"][1] + (a["end"][1] - a["start"][1]) * t
        ball = (bx, by)
        if a["type"] != "Dribble":
            col = "white" if a["type"] == "Pass" else ("#9db4c9" if a["type"] == "Carry" else ORANGE)
            ax.plot([a["start"][0], bx], [a["start"][1], by], color=col, linewidth=3,
                    linestyle="-" if a["type"] != "Carry" else "--", zorder=2)
    for pl in order_players:
        i = player_idx[pl]
        if first_action[pl] <= last_vis:
            ax.scatter([xs_p[i]], [ys_p[i]], s=330, c=BLUE, edgecolors="white", linewidths=1.8, zorder=3)
            lx, ly, leader = label_pos[i]
            if leader:
                ax.plot([xs_p[i], lx], [ys_p[i], ly], color="#c9d6e2", linewidth=1, alpha=0.7, zorder=1.5)
            tx = ax.text(lx, ly, disp[pl], ha="center", va="center", color="white",
                         fontsize=11.5, fontweight="bold", zorder=4)
            tx.set_path_effects([pe.withStroke(linewidth=3, foreground=BG)])
    if ball is not None:
        ax.scatter([ball[0]], [ball[1]], s=220, c="white", edgecolors=BG, linewidths=2, zorder=6)
    line1_t.set_text(line1)
    involved = [disp[pl] for pl in order_players if first_action[pl] <= last_vis]
    chain_t1.set_text("  \u2192  ".join(involved[:4]) if rows_vis == 0 else "")
    chain_t2.set_text("  \u2192  ".join(involved[4:8]) if rows_vis == 0 and len(involved) > 4 else "")
    caption_t.set_visible(rows_vis > 0)
    for k, tt in enumerate(table_texts):
        tt.set_visible(k < rows_vis)


# ---------- frame schedule ----------
n_act = len(actions)
FPA = int(np.clip(150 // max(n_act, 1), 6, 12))
frames = []
seen = set()
for k, a in enumerate(actions):
    n_f = FPA + (4 if a["type"] == "Shot" else 0)
    seen.add(a["player"])
    for f in range(n_f):
        t = (f + 1) / n_f
        t = t * t * (3 - 2 * t)
        frames.append((k, k, t, f"Players involved so far: {len(seen)}", 0))
    frames += [(k + 1, None, 0, f"Players involved so far: {len(seen)}", 0)] * 2
frames += [(n_act, None, 0, f"GOAL   |   shot xG {goal_xg:.2f}", 0)] * 18
for rv in range(1, n_rows + 2):
    frames += [(n_act, None, 0, "", rv)] * 5
frames += [(n_act, None, 0, "", n_rows + 1)] * 55

# static preview of the finished credit table
draw_frame(n_act, None, 0, "", n_rows + 1)
fig.savefig("goal_move_credit.png", dpi=108, facecolor=BG)
print("Saved chart: goal_move_credit.png")


def update(i):
    draw_frame(*frames[i])


FPS = 20
RAW_VIDEO = "goal_move_replay_raw.mp4"
FINAL_VIDEO = "goal_move_replay.mp4"
anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / FPS)
writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=4000, extra_args=["-pix_fmt", "yuv420p"])
print(f"Rendering {len(frames)} frames ({len(frames) / FPS:.0f} seconds), this takes a minute...")
anim.save(RAW_VIDEO, writer=writer, dpi=108, savefig_kwargs={"facecolor": BG})
print("Saved replay:", RAW_VIDEO)


# ---------- append the charts to the video as end cards ----------
def append_end_cards(video_in, video_out, cards, seconds, size=1080, fps=20, bg="0d1b2a"):
    cards = [c for c in cards if os.path.exists(c)]
    if not cards:
        print("No chart PNGs found, keeping the replay without end cards.")
        os.replace(video_in, video_out)
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-i", video_in]
    for c in cards:
        cmd += ["-loop", "1", "-framerate", str(fps), "-t", str(seconds), "-i", c]
    norm = (f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
            f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color=0x{bg},"
            f"setsar=1,fps={fps},format=yuv420p")
    n = len(cards) + 1
    parts = [f"[{i}:v]{norm}[v{i}]" for i in range(n)]
    labels = "".join(f"[v{i}]" for i in range(n))
    fc = ";".join(parts) + f";{labels}concat=n={n}:v=1:a=0[out]"
    cmd += ["-filter_complex", fc, "-map", "[out]", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart", video_out]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-1500:])
        print("End cards failed, keeping the replay without them.")
        os.replace(video_in, video_out)
        return
    os.remove(video_in)
    print("Added end cards:", ", ".join(cards))


append_end_cards(RAW_VIDEO, FINAL_VIDEO, END_CARDS, END_CARD_SECONDS, fps=FPS)
print("Saved video:", FINAL_VIDEO)