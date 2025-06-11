import re
import requests
import pandas as pd
import time
from collections import Counter
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations



def find_player_ids(starting_names, bench_names, unique_players):
    found = []
    all_names = starting_names + bench_names

    # Check for duplicate names in user input
    name_counts = Counter(all_names)
    duplicates = [name for name, count in name_counts.items() if count > 1]
    if duplicates:
        print(f"Duplicate player(s) in selection: {', '.join(duplicates)}")
        return None, None

    # Match each name to a player
    for name in all_names:
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        matched_rows = unique_players[unique_players['full_name'].str.contains(pattern)]
        if not matched_rows.empty:
            for _, row in matched_rows.iterrows():
                found.append((
                    row['player_id'], 
                    row['full_name'], 
                    row['team_name'], 
                    row['position']
                ))
        else:
            found.append((None, name, None, None))

    # Print any unmatched players
    for pid, pname, _, _ in found:
        if pid is None:
            print(f"Player not found: {pname}")

    # Team constraint: max 3 per team
    team_counter = Counter(team for pid, _, team, _ in found if pid is not None)
    if any(count > 3 for count in team_counter.values()):
        print("Please select at most 3 players from each team.")
        return None, None

    # Position constraint for starting XI: 1 GK, at least 3 DEF, at least 2 MID, at least 1 FWD, 11 total
    starting_positions = [pos for i, (pid, _, _, pos) in enumerate(found) if pid is not None and i < len(starting_names)]
    bench_positions = [pos for i, (pid, _, _, pos) in enumerate(found) if pid is not None and i >= len(starting_names)]

    required_start = {'Goalkeeper': 1, 'Defender': 3, 'Midfielder': 2, 'Forward': 1}
    start_counter = Counter(starting_positions)
    if start_counter['Goalkeeper'] != 1 or start_counter['Defender'] < 3 or start_counter['Midfielder'] < 2 or start_counter['Forward'] < 1 or len(starting_positions) != 11:
        print("Starting XI must have exactly 1 Goalkeeper, at least 3 Defenders, at least 2 Midfielders, at least 1 Forward, and 11 players in total.")
        print("Your starting XI selection:", dict(start_counter))
        return None, None

    if len(bench_positions) != 4:
        print("Bench must have exactly 4 players.")
        return None, None

    # Check for duplicate player IDs (in case of ambiguous name matches)
    player_ids = [pid for pid, _, _, _ in found if pid is not None]
    pid_counts = Counter(player_ids)
    pid_duplicates = [str(pid) for pid, count in pid_counts.items() if count > 1]
    if pid_duplicates:
        print(f"Duplicate player(s) detected by ID: {', '.join(pid_duplicates)}")
        return None, None

    # Prepare DataFrames for starting XI and bench
    import pandas as pd
    starting_data = [found[i] for i in range(len(starting_names))]
    bench_data = [found[i] for i in range(len(starting_names), len(found))]

    starting_xi = pd.DataFrame(starting_data, columns=['player_id', 'full_name', 'team_name', 'position'])
    bench = pd.DataFrame(bench_data, columns=['player_id', 'full_name', 'team_name', 'position'])

    return starting_xi, bench
def recommend_captain(fpl_data_pred, user_starting_xi_df, unique_players, current_gw):
    """
    Recommend the best captain from the starting XI based on aggregated predicted points for the next gameweek,
    handling double gameweeks by summing predicted points.

    Parameters:
    - fpl_data_pred: dataframe containing 'player_id', 'round', 'next_gw_pred'
    - user_starting_xi_df: DataFrame with at least a 'player_id' column (11 players)
    - unique_players: dataframe with columns 'player_id' and 'full_name'
    - current_gw: int, current gameweek number

    Returns:
    - captain_id: player_id of recommended captain
    - captain_name: full_name of recommended captain
    - predicted_points: aggregated predicted points for the captain
    """

    next_gw = current_gw

    # Accept DataFrame input for user_starting_xi
    player_ids = user_starting_xi_df['player_id'].tolist()

    # Filter for next gameweek and user's starting XI players
    starters_next_gw = fpl_data_pred[
        (fpl_data_pred['round'] == next_gw) &
        (fpl_data_pred['player_id'].isin(player_ids))
    ].copy()

    if starters_next_gw.empty:
        raise ValueError(f"No data available for next gameweek {next_gw} and given starting XI.")

    # Aggregate predicted points by player_id (sum to handle double gameweeks)
    agg_preds = starters_next_gw.groupby('player_id', as_index=False)['next_gw_pred'].sum()

    # Find player with max aggregated predicted points
    best_captain_row = agg_preds.loc[agg_preds['next_gw_pred'].idxmax()]
    captain_id = best_captain_row['player_id']
    predicted_points = best_captain_row['next_gw_pred']

    # Get captain name from unique_players
    captain_name = unique_players.loc[unique_players['player_id'] == captain_id, 'full_name'].values[0]

    return captain_id, captain_name, predicted_points
def select_best_starting_xi_and_bench(fpl_data_pred, starting_xi_df, bench_df, current_gw, unique_players):
    import pandas as pd

    next_gw = current_gw
    all_ids = list(starting_xi_df['player_id']) + list(bench_df['player_id'])
    if len(all_ids) != 15:
        raise ValueError("You must provide exactly 15 unique player_ids (11 starting XI + 4 bench).")

    # Filter data for next GW and squad players
    squad_next_gw = fpl_data_pred[
        (fpl_data_pred['player_id'].isin(all_ids)) & (fpl_data_pred['round'] == next_gw)
    ].copy()

    # Aggregate predicted points if multiple rows per player
    agg_df = squad_next_gw.groupby(['player_id', 'position'], as_index=False).agg({'next_gw_pred': 'sum'})

    # Merge to get full_name and team_name from unique_players
    agg_df = agg_df.merge(
        unique_players[['player_id', 'full_name', 'team_name']],
        on='player_id',
        how='left'
    )

    # --- FILL IN missing players with next_gw_pred = 0 ---
    missing_ids = set(all_ids) - set(agg_df['player_id'])
    if missing_ids:
        missing_rows = unique_players[unique_players['player_id'].isin(missing_ids)].copy()
        missing_rows['next_gw_pred'] = 0
        agg_df = pd.concat([agg_df, missing_rows[['player_id', 'position', 'next_gw_pred', 'full_name', 'team_name']]], ignore_index=True)

    # Sort by predicted points descending
    agg_df = agg_df.sort_values('next_gw_pred', ascending=False).reset_index(drop=True)

    # Pick the best GK (exactly 1)
    gk_df = agg_df[agg_df['position'] == 'Goalkeeper']
    if len(gk_df) == 0:
        raise ValueError("No goalkeeper in squad for next gameweek.")
    best_gk = gk_df.iloc[0:1]

    # Pick at least 3 defenders, sorted by points
    def_df = agg_df[agg_df['position'] == 'Defender']
    if len(def_df) < 3:
        raise ValueError("Not enough defenders in squad for next gameweek.")
    best_defenders = def_df.iloc[0:3]

    # Exclude chosen GK and defenders from remaining pool
    exclude_ids = list(best_gk['player_id']) + list(best_defenders['player_id'])
    remaining_pool = agg_df[~agg_df['player_id'].isin(exclude_ids)]

    # We have selected 1 GK + 3 Defenders = 4 players so far
    slots_left = 11 - 4

    # Exclude any goalkeepers from remaining picks (to avoid second GK)
    remaining_pool = remaining_pool[remaining_pool['position'] != 'Goalkeeper']

    # Pick top remaining players (any position except GK) by predicted points
    best_remaining = remaining_pool.head(slots_left)

    # Compose optimal starting XI
    optimal_starting_11 = pd.concat([best_gk, best_defenders, best_remaining])
    optimal_starting_11 = optimal_starting_11.sort_values('next_gw_pred', ascending=False).reset_index(drop=True)

    if len(optimal_starting_11) != 11:
        raise ValueError(f"Starting XI size error: expected 11, got {len(optimal_starting_11)}")

    # Confirm position constraints again
    pos_counts = optimal_starting_11['position'].value_counts()
    if pos_counts.get('Goalkeeper', 0) != 1:
        raise ValueError(f"Starting XI must have exactly 1 goalkeeper, got {pos_counts.get('Goalkeeper', 0)}")
    if pos_counts.get('Defender', 0) < 3:
        raise ValueError(f"Starting XI must have at least 3 defenders, got {pos_counts.get('Defender', 0)}")

    # Remaining 4 players are optimal bench
    optimal_bench = agg_df[~agg_df['player_id'].isin(optimal_starting_11['player_id'])]
    optimal_bench = optimal_bench.sort_values('next_gw_pred', ascending=False).reset_index(drop=True)

    if len(optimal_bench) != 4:
        raise ValueError(f"Bench size error: expected 4, got {len(optimal_bench)}")

    # Drop next_gw_pred before returning (optional)
    cols_to_keep = ['player_id', 'position', 'full_name', 'team_name']
    optimal_starting_11 = optimal_starting_11[cols_to_keep]
    optimal_bench = optimal_bench[cols_to_keep]

    return optimal_starting_11, optimal_bench
def plot_mplsoccer_pitch_starting11_and_bench(starting_xi_df, bench_df, fpl_data_pred, current_gw, captain_name=None):
    """
    Plots the starting 11 and bench on a vertical soccer pitch.
    Under each player, their predicted points for the current gameweek are shown.
    Doubles captain's predicted points if captain_name is specified.

    Args:
        starting_xi_df (pd.DataFrame): Starting 11 players.
        bench_df (pd.DataFrame): Bench players.
        fpl_data_pred (pd.DataFrame): Contains player_id, next_gw_pred, and value.
        current_gw (int): Current gameweek.
        captain_name (str): Full name of the captain (must match one of the starting XI).
    
    Returns:
        total_points_xi (float): Total predicted points for Starting XI (captain doubled)
        total_team_value (float): Total value (£M) of Starting XI + Bench
    """
    from mplsoccer import VerticalPitch
    import matplotlib.pyplot as plt
    import pandas as pd

    # Merge predicted points and value
    gw_preds = fpl_data_pred[fpl_data_pred['round'] == current_gw][['player_id', 'next_gw_pred', 'value']]
    starting_xi_df = starting_xi_df.merge(gw_preds, on='player_id', how='left').fillna({'next_gw_pred': 0, 'value': 0})
    bench_df = bench_df.merge(gw_preds, on='player_id', how='left').fillna({'next_gw_pred': 0, 'value': 0})

    # Double captain's points if specified
    if captain_name and captain_name in starting_xi_df['full_name'].values:
        starting_xi_df.loc[starting_xi_df['full_name'] == captain_name, 'next_gw_pred'] *= 2

    # Calculate totals
    total_points_xi = starting_xi_df['next_gw_pred'].sum()
    total_team_value = (starting_xi_df['value'].sum() + bench_df['value'].sum()) / 10  # Convert to £M

    # Position mapping
    pos_map = {'Goalkeeper': 'GKP', 'Defender': 'DEF', 'Midfielder': 'MID', 'Forward': 'FWD'}

    # Prepare starting XI
    players = []
    for _, row in starting_xi_df.iterrows():
        players.append({
            'name': row['full_name'],
            'position': pos_map[row['position']],
            'points': row['next_gw_pred']
        })

    # Prepare bench
    bench_players = []
    bench_gk = bench_df[bench_df['position'] == 'Goalkeeper']
    bench_outfield = bench_df[bench_df['position'] != 'Goalkeeper']
    for _, row in pd.concat([bench_gk, bench_outfield]).iterrows():
        bench_players.append({
            'name': row['full_name'],
            'position': pos_map[row['position']],
            'points': row['next_gw_pred']
        })

    # Count per position
    n_def = sum(1 for p in players if p['position'] == 'DEF')
    n_mid = sum(1 for p in players if p['position'] == 'MID')
    n_fwd = sum(1 for p in players if p['position'] == 'FWD')

    # Adjust x_positions
    y_positions = {'GKP': 10, 'DEF': 20, 'MID': 40, 'FWD': 60}
    x_positions = {
        'GKP': [50],
        'DEF': [87, 71, 50, 29, 7],
        'MID': [87, 71, 50, 29, 7],
        'FWD': [80, 50, 20]
    }
    if n_def == 4:
        x_positions['DEF'] = [85, 62, 38, 15]
    elif n_def == 3:
        x_positions['DEF'] = [80, 50, 20]
    if n_mid == 4:
        x_positions['MID'] = [85, 62, 38, 15]
    elif n_mid == 3:
        x_positions['MID'] = [80, 50, 20]
    if n_fwd == 2:
        x_positions['FWD'] = [35, 65]
    elif n_fwd == 1:
        x_positions['FWD'] = [50]

    colors = {'GKP': 'blue', 'DEF': 'green', 'MID': 'orange', 'FWD': 'red'}

    # Group by position and track points
    pos_groups = {'GKP': [], 'DEF': [], 'MID': [], 'FWD': []}
    points_map = {}
    for player in players:
        pos_groups[player['position']].append(player['name'])
        points_map[player['name']] = player['points']

    fig, ax = plt.subplots(figsize=(5, 9))
    pitch = VerticalPitch(pitch_type='opta', pitch_color='#262730', line_color='white')
    pitch.draw(ax=ax)

    # Plot Starting XI
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        names = pos_groups[pos]
        xs = x_positions[pos]
        base_y = y_positions[pos]
        y_offsets = [0] * len(names)
        if pos in ['DEF', 'MID', 'FWD']:
            if len(names) == 5:
                y_offsets = [0, 5, 0, 5, 0]
            elif len(names) == 4:
                y_offsets = [0, 5, 0, 5]
            elif len(names) == 3:
                y_offsets = [0, 5, 0]

        for i, name in enumerate(names):
            if i < len(xs):
                x = xs[i]
                y = 100 - (base_y + y_offsets[i])
                is_captain = name == captain_name
                player_points = points_map[name]
                short_name = name[:15]  # Truncate to 15 characters
                label = f"{short_name}{' (C)' if is_captain else ''}\n({points_map[name]:.1f}pts)"
                pitch.annotate(
                    label,
                    xy=(y, x),
                    xytext=(0, 0),
                    textcoords='offset points',
                    ha='center',
                    va='center',
                    fontsize=7,
                    fontweight='bold',
                    color='white',
                    ax=ax,
                    bbox=dict(boxstyle='round,pad=0.2', fc='#0E1117', ec=colors[pos], lw=1, alpha=0.8)
                )


    # Labels with subtext
    ax.text(50, 20, "Bench Players\n(Predicted Pts)", fontsize=9.5, fontweight='bold', color='black',
            ha='center', va='center', alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='black', lw=2, alpha=0.7))

    ax.text(50, 98, "Starting XI\n(Predicted Pts)", fontsize=9.5, fontweight='bold', color='black',
            ha='center', va='center', alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='black', lw=2, alpha=0.7))


    # Plot Bench
    bench_y = 95
    bench_xs = [85, 62, 38, 15]
    bench_y_offsets = [0, 4, 0, 4] if len(bench_players) == 4 else [0] * len(bench_players)
    for i, player in enumerate(bench_players):
        if i < len(bench_xs):
            y = 100 - (bench_y + bench_y_offsets[i]) + 5
            short_name = player['name'][:15]
            name_line = f"{short_name}\n({player['points']:.1f}pts)"
            pitch.annotate(
                name_line,
                xy=(y, bench_xs[i]),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center',
                va='center',
                fontsize=7,
                fontweight='bold',
                color='white',
                ax=ax,
                bbox=dict(boxstyle='round,pad=0.2', fc='#0E1117', ec=colors.get(player['position'], 'black'), lw=1, alpha=0.8)
            )

    return total_points_xi, total_team_value


def optimize_squad_transfers_topk(
    fpl_data_pred,
    starting_xi_df,
    bench_df,
    current_gw,
    unique_players,
    transfers_allowed=1,
    total_value_limit=1000,
    topk_out=6,
    topk_in=20,
):
    """
    Optimize FPL squad by swapping up to transfers_allowed players,
    but only consider bottom topk_out current players for transfer out
    and top topk_in players outside squad for transfer in.

    Returns:
      new_optimised_xi: DataFrame with 11 players for starting XI
      new_optimised_bench: DataFrame with 4 players on bench
      transfers_summary: dict with 'transfers_in' and 'transfers_out' (player names)
    """

    # Combine current squad
    current_squad = pd.concat([starting_xi_df, bench_df], ignore_index=True)
    current_player_ids = set(current_squad['player_id'])

    # Filter prediction for next GW and merge player info
    pred_next_gw = fpl_data_pred[
        (fpl_data_pred['round'] == current_gw) &
        (fpl_data_pred['player_id'].isin(unique_players['player_id']))
    ][['player_id', 'next_gw_pred', 'value']].drop_duplicates(subset='player_id')

    pred_next_gw = pred_next_gw.merge(
        unique_players[['player_id', 'position', 'team_name', 'full_name']],
        on='player_id', how='left'
    )

    # Create dict for quick lookup
    player_info = pred_next_gw.set_index('player_id').to_dict('index')

    # Current players info
    current_players_df = pred_next_gw[pred_next_gw['player_id'].isin(current_player_ids)].copy()

    # Select bottom topk_out players by predicted points from current squad (worst performers)
    bottom_current = current_players_df.nsmallest(topk_out, 'next_gw_pred')['player_id'].tolist()

    # Potential players to bring in (outside current squad)
    potential_ins_all = set(unique_players['player_id']) - current_player_ids
    potential_ins_df = pred_next_gw[pred_next_gw['player_id'].isin(potential_ins_all)].copy()

    # Filter to those with predicted points > 1
    potential_ins_df = potential_ins_df[potential_ins_df['next_gw_pred'] > 1]

    # Select top topk_in players by predicted points
    top_potential_ins = potential_ins_df.nlargest(topk_in, 'next_gw_pred')['player_id'].tolist()

    # --- Helper functions (same as your original code) ---

    def valid_squad(player_ids):
        if len(player_ids) != 15:
            return False

        pos_counts = {'Goalkeeper':0, 'Defender':0, 'Midfielder':0, 'Forward':0}
        team_counts = {}

        for pid in player_ids:
            p = player_info.get(pid)
            if p is None:
                return False
            pos_counts[p['position']] += 1
            team_counts[p['team_name']] = team_counts.get(p['team_name'], 0) + 1
            if team_counts[p['team_name']] > 3:
                return False

        if pos_counts['Goalkeeper'] != 2:
            return False
        if pos_counts['Defender'] != 5:
            return False
        if pos_counts['Midfielder'] != 5:
            return False
        if pos_counts['Forward'] != 3:
            return False

        total_val = sum(player_info[pid]['value'] for pid in player_ids)
        if total_val > total_value_limit:
            return False

        return True

    def valid_starting_xi(player_ids):
        pos_counts = {'Goalkeeper':0, 'Defender':0, 'Midfielder':0, 'Forward':0}
        team_counts = {}

        for pid in player_ids:
            p = player_info.get(pid)
            if p is None:
                return False
            pos_counts[p['position']] += 1
            team_counts[p['team_name']] = team_counts.get(p['team_name'], 0) + 1
            if team_counts[p['team_name']] > 3:
                return False

        if pos_counts['Goalkeeper'] != 1:
            return False
        if pos_counts['Defender'] < 3:
            return False

        outfield = pos_counts['Defender'] + pos_counts['Midfielder'] + pos_counts['Forward']
        if outfield != 10:
            return False

        valid_formations = [
            (3, 4, 3), (3, 5, 2),
            (4, 4, 2), (4, 3, 3),
            (5, 3, 2), (5, 4, 1)
        ]
        formation = (pos_counts['Defender'], pos_counts['Midfielder'], pos_counts['Forward'])
        if formation not in valid_formations:
            return False

        return True

    def total_pred_points(player_ids):
        return sum(player_info[pid]['next_gw_pred'] for pid in player_ids)

    def pick_best_starting_xi(squad_ids):
        best_xi = None
        best_pts = -float('inf')

        players_by_pos = {'Goalkeeper':[], 'Defender':[], 'Midfielder':[], 'Forward':[]}
        for pid in squad_ids:
            pos = player_info[pid]['position']
            players_by_pos[pos].append(pid)

        gk_candidates = players_by_pos['Goalkeeper']
        if len(gk_candidates) < 2:
            return None, None

        for gk in gk_candidates:
            bench_gk = [pid for pid in gk_candidates if pid != gk][0]

            def_candidates = players_by_pos['Defender']
            mid_candidates = players_by_pos['Midfielder']
            fwd_candidates = players_by_pos['Forward']

            for D_req, M_req, F_req in [
                (3, 4, 3), (3, 5, 2),
                (4, 4, 2), (4, 3, 3),
                (5, 3, 2), (5, 4, 1)
            ]:
                if len(def_candidates) < D_req or len(mid_candidates) < M_req or len(fwd_candidates) < F_req:
                    continue

                def_combos = combinations(def_candidates, D_req)
                mid_combos = combinations(mid_candidates, M_req)
                fwd_combos = combinations(fwd_candidates, F_req)

                mid_combos = list(mid_combos)
                fwd_combos = list(fwd_combos)

                for d_combo in def_combos:
                    for m_combo in mid_combos:
                        for f_combo in fwd_combos:
                            xi_ids = [gk] + list(d_combo) + list(m_combo) + list(f_combo)

                            team_counts = {}
                            valid = True
                            for pid in xi_ids:
                                team = player_info[pid]['team_name']
                                team_counts[team] = team_counts.get(team, 0) + 1
                                if team_counts[team] > 3:
                                    valid = False
                                    break
                            if not valid:
                                continue

                            pts = total_pred_points(xi_ids)
                            if pts > best_pts:
                                best_pts = pts
                                best_xi = xi_ids

        if best_xi is None:
            return None, None

        bench_ids = set(squad_ids) - set(best_xi)
        return list(best_xi), list(bench_ids)

    # Main optimization loop
    current_squad_list = list(current_player_ids)

    best_score = -float('inf')
    best_squad = None
    best_starting_xi = None
    best_bench = None
    best_out = []
    best_in = []

    for k in range(0, transfers_allowed + 1):
        # When k=0: no transfer, just check current squad
        if k == 0:
            if valid_squad(current_player_ids):
                xi_ids, bench_ids = pick_best_starting_xi(current_player_ids)
                if xi_ids is not None:
                    total_pts = total_pred_points(xi_ids)
                    if total_pts > best_score:
                        best_score = total_pts
                        best_squad = current_player_ids
                        best_starting_xi = xi_ids
                        best_bench = bench_ids
                        best_out = []
                        best_in = []
            continue

        # For k > 0, consider combos
        outs_combos = combinations(bottom_current, k)
        ins_combos = list(combinations(top_potential_ins, k))

        for out_ids in outs_combos:
            out_set = set(out_ids)
            remaining_players = set(current_squad_list) - out_set

            for in_ids in ins_combos:
                in_set = set(in_ids)

                candidate_squad = remaining_players | in_set

                if not valid_squad(candidate_squad):
                    continue

                xi_ids, bench_ids = pick_best_starting_xi(candidate_squad)
                if xi_ids is None:
                    continue

                total_pts = total_pred_points(xi_ids) # Transfer penalty

                if total_pts > best_score:
                    best_score = total_pts
                    best_squad = candidate_squad
                    best_starting_xi = xi_ids
                    best_bench = bench_ids
                    best_out = out_ids
                    best_in = in_ids

    # Prepare output DataFrames
    if best_squad is None:
        # fallback: return original
        return starting_xi_df, bench_df, {'transfers_in': [], 'transfers_out': []}

    def build_df(player_ids):
        data = []
        for pid in player_ids:
            p = player_info[pid]
            data.append({
                'player_id': pid,
                'full_name': p['full_name'],
                'position': p['position'],
                'team_name': p['team_name'],
                'pred_points': p['next_gw_pred'],
                'value': p['value']
            })
        return pd.DataFrame(data)

    df_xi = build_df(best_starting_xi)
    df_bench = build_df(best_bench)

    transfers_summary = {
        'transfers_in': [player_info[pid]['full_name'] for pid in best_in],
        'transfers_out': [player_info[pid]['full_name'] for pid in best_out]
    }

    return df_xi.reset_index(drop=True), df_bench.reset_index(drop=True), transfers_summary


def fill_missing_gameweeks(fpl_df):
    all_gameweeks = set(range(1, 39))
    info_cols = [
        'element', 'fixture', 'opponentteam_id', 'total_points', 'was_home',
        'kickoff_time', 'team_h_score', 'team_a_score', 'modified', 'minutes',
        'goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 'own_goals',
        'penalties_saved', 'penalties_missed', 'yellow_cards', 'red_cards', 'saves',
        'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index', 'starts',
        'expected_goals', 'expected_assists', 'expected_goal_involvements',
        'expected_goals_conceded', 'mng_win', 'mng_draw', 'mng_loss',
        'mng_underdog_win', 'mng_underdog_draw', 'mng_clean_sheets', 'mng_goals_scored',
        'value', 'transfers_balance', 'selected', 'transfers_in', 'transfers_out',
        'first_name', 'second_name', 'team_name', 'position', 'opponent_team_name',
        'was_home_opponent', 'fdr', 'position_encoded', 'fixture_order', 'round_adjusted',
        'master_id', 'player_id', 'id'
    ]

    new_rows = []

    for player_id, player_df in fpl_df.groupby('player_id'):
        played_rounds = set(player_df['round'].unique())
        missing_rounds = all_gameweeks - played_rounds

        if missing_rounds:
            # Get player info from first row for this player
            player_info = player_df.iloc[0][info_cols].to_dict()

            for r in missing_rounds:
                new_row = player_info.copy()
                new_row['round'] = r
                new_row['next_gw_pred'] = 0
                new_rows.append(new_row)

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        fpl_df_filled = pd.concat([fpl_df, df_new], ignore_index=True)
        # Optional: sort by player and round
        fpl_df_filled = fpl_df_filled.sort_values(['player_id', 'round']).reset_index(drop=True)
        return fpl_df_filled
    else:
        return fpl_df
