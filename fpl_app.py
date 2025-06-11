import streamlit as st
import pandas as pd
from function_utils import *
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
from PIL import Image

# Load and display logo in the top-left
logo = Image.open("logo.png")

col1, col2 = st.columns([1, 6])
with col1:
    st.image(logo, width=80)  # Adjust width as needed
st.markdown("<small>Created by Ibrahim Oksuzoglu</small>", unsafe_allow_html=True)


# Load data
@st.cache_data
def load_data():
    unique_players = pd.read_pickle('unique_players.pkl')
    fpl_data_pred = pd.read_pickle('fpl_data_pred.pkl')
    return unique_players, fpl_data_pred

unique_players, fpl_data_pred = load_data()
fpl_data_pred = fill_missing_gameweeks(fpl_data_pred)

# Filter out managers
fpl_data_pred = fpl_data_pred[fpl_data_pred['position'] != 'Manager']
unique_players = unique_players[unique_players['position'] != 'Manager']
all_players = sorted(unique_players['full_name'].dropna().unique())

# Style multiselect to be scrollable
st.markdown("""
    <style>
    .stMultiSelect > div {
        max-height: 150px;
        overflow-y: auto;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h3 style="display: flex; align-items: center; text-align: center; font-size: 36px;">
        <span style="flex: 1; height: 3px; background: #ccc;"></span>
        <span style="padding: 0 10px;">⚽ Fantasy Football Optimiser</span>
        <span style="flex: 1; height: 3px; background: #ccc;"></span>
    </h3>
""", unsafe_allow_html=True)



# Autofill checkbox
autofill = st.checkbox("👈 Autofill with Preset Team")

example_starting_names = [
    "Matz Sels", "William Saliba", "Neco Williams", "Joško Gvardiol", "Bryan Mbeumo",
    "Morgan Rogers", "Tino Livramento", "Jarrod Bowen", "Yoane Wissa", "Chris Wood", "Daniel Muñoz"
]
example_bench_names = ["David Raya Martin", "Ollie Watkins", "Mohamed Salah", "Jacob Murphy"]

# Player selection
col1_select, col2_select, col3_select = st.columns(3)

with col1_select:
    starting_names = st.multiselect(
        "👥 Select Your Starting XI (11 players)",
        options=[p for p in all_players if p not in st.session_state.get("bench", [])],
        default=example_starting_names if autofill else [],
        key="starting_names"
    )

with col2_select:
    bench_names = st.multiselect(
        "🪑 Select Your Bench (4 players)",
        options=[p for p in all_players if p not in st.session_state.get("starting_names", [])],
        default=example_bench_names if autofill else [],
        key="bench"
    )

with col3_select:
    captain_name = st.selectbox("🧢 Select Your Captain", options=starting_names)
    transfers_allowed = st.selectbox("🔄 Select Transfer Amount", options=list(range(0, 6)), index=3)

# Gameweek & plot button
current_gw = st.slider("📅 Select Desired Gameweek", min_value=1, max_value=37, value=25, step=1)
plot_button = st.button("🤖 Run Prediction and Optimisation Model")

# Validation & main logic
if plot_button:
    if len(starting_names) != 11:
        st.warning("⚠️ Please select exactly 11 players for your Starting XI.")
    if len(bench_names) != 4:
        st.warning("⚠️ Please select exactly 4 players for your Bench.")
    if set(starting_names) & set(bench_names):
        st.error("❌ A player cannot be in both Starting XI and Bench.")

    if (
        len(starting_names) == 11 and
        len(bench_names) == 4 and
        not (set(starting_names) & set(bench_names))
    ):
        starting_players_df = unique_players[unique_players['full_name'].isin(starting_names)]
        bench_players_df = unique_players[unique_players['full_name'].isin(bench_names)]

        num_gk_starting = (starting_players_df['position'] == 'Goalkeeper').sum()
        num_def_starting = (starting_players_df['position'] == 'Defender').sum()
        num_mid_starting = (starting_players_df['position'] == 'Midfielder').sum()
        num_fwd_starting = (starting_players_df['position'] == 'Forward').sum()

        num_gk_bench = (bench_players_df['position'] == 'Goalkeeper').sum()
        num_def_bench = (bench_players_df['position'] == 'Defender').sum()
        num_mid_bench = (bench_players_df['position'] == 'Midfielder').sum()
        num_fwd_bench = (bench_players_df['position'] == 'Forward').sum()

        valid = True

        combined_teams = list(starting_players_df['team_name']) + list(bench_players_df['team_name'])
        combined_team_counts = pd.Series(combined_teams).value_counts()
        if any(combined_team_counts > 3):
            teams_exceeding = combined_team_counts[combined_team_counts > 3].index.tolist()
            st.error(f"🚫 Cannot have more than 3 players from the same team: {', '.join(teams_exceeding)}")
            valid = False

        if num_gk_starting != 1:
            st.error(f"🧤 Must have exactly 1 Goalkeeper in Starting XI. You have: {num_gk_starting}")
            valid = False
        if not (3 <= num_def_starting <= 5):
            st.error(f"🛡️ Must have 3–5 Defenders in Starting XI. You have: {num_def_starting}")
            valid = False
        if not (2 <= num_mid_starting <= 5):
            st.error(f"🎯 Must have 2–5 Midfielders in Starting XI. You have: {num_mid_starting}")
            valid = False
        if not (1 <= num_fwd_starting <= 3):
            st.error(f"⚔️ Must have 1–3 Forwards in Starting XI. You have: {num_fwd_starting}")
            valid = False
        if num_gk_bench != 1:
            st.error(f"🧤 Must have exactly 1 Goalkeeper on the Bench. You have: {num_gk_bench}")
            valid = False

        if valid:
            starting_xi, bench = find_player_ids(starting_names, bench_names, unique_players)
            optimal_starting_11, optimal_bench = select_best_starting_xi_and_bench(
                fpl_data_pred, starting_xi, bench, current_gw, unique_players
            )
            new_optimised_xi, new_optimised_bench, transfer_summary = optimize_squad_transfers_topk(
                fpl_data_pred,
                optimal_starting_11,
                optimal_bench,
                current_gw,
                unique_players,
                transfers_allowed,
                total_value_limit=1000,
                topk_out=6,
                topk_in=20,
            )
            rec_captain_id, rec_captain_name, rec_predicted_points = recommend_captain(
                fpl_data_pred,
                new_optimised_xi.drop(columns=['pred_points','value']),
                unique_players,
                current_gw
            )

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                st.markdown("""
                    <h3 style="display: flex; align-items: center; text-align: center;">
                        <span style="flex: 1; height: 3px; background: #ccc;"></span>
                        <span style="padding: 0 10px;">🎲 Your Team</span>
                        <span style="flex: 1; height: 3px; background: #ccc;"></span>
                    </h3>
                """, unsafe_allow_html=True)
                original_total_points_xi, original_total_team_value = plot_mplsoccer_pitch_starting11_and_bench(
                    starting_xi, bench, fpl_data_pred, current_gw, captain_name
                )
                st.pyplot(plt.gcf(), use_container_width=False)

            with col3:
                st.markdown("""
                    <h3 style="display: flex; align-items: center; text-align: center;">
                        <span style="flex: 1; height: 3px; background: #ccc;"></span>
                        <span style="padding: 0 10px;">🧠 Optimised Team</span>
                        <span style="flex: 1; height: 3px; background: #ccc;"></span>
                    </h3>
                """, unsafe_allow_html=True)
                optimised_total_points_xi, optimised_total_team_value = plot_mplsoccer_pitch_starting11_and_bench(
                    new_optimised_xi.drop(columns=['pred_points','value']),
                    new_optimised_bench.drop(columns=['pred_points','value']),
                    fpl_data_pred,
                    current_gw,
                    rec_captain_name
                )
                st.pyplot(plt.gcf(), use_container_width=False)

            with col2:
                st.markdown("""
                    <h3 style="display: flex; align-items: center; text-align: center;">
                        <span style="flex: 1; height: 3px; background: #ccc;"></span>
                        <span style="padding: 0 10px;">🔁 Recommended Team Changes</span>
                        <span style="flex: 1; height: 3px; background: #ccc;"></span>
                    </h3>
                """, unsafe_allow_html=True)
                points_gained = optimised_total_points_xi - original_total_points_xi
                st.markdown(f"<h3 style='text-align: center;'>🏆 Points Gained: {points_gained:.2f}</h3>", unsafe_allow_html=True)

                if transfer_summary:
                    out_names = transfer_summary.get('transfers_out', [])
                    in_names = transfer_summary.get('transfers_in', [])
                    max_len = max(len(out_names), len(in_names))
                    out_names += [""] * (max_len - len(out_names))
                    in_names += [""] * (max_len - len(in_names))

                    col_out, col_in = st.columns(2)
                    with col_out:
                        st.markdown("### ❌ Transfers Out")
                        for name in out_names:
                            st.markdown(f"- {name}")
                    with col_in:
                        st.markdown("### ✅ Transfers In")
                        for name in in_names:
                            st.markdown(f"- {name}")
                else:
                    st.info("🟢 No transfers made. Your current squad is already optimal.")

                left_col, right_col = st.columns(2)
                with left_col:
                    st.metric("📉 Your Predicted Points (XI)", f"{original_total_points_xi:.2f}")
                    st.metric("💰 Your Team Value (£M)", f"{original_total_team_value:.1f}")
                    st.metric("🧢 Your Captain", f"{captain_name}")

                with right_col:
                    st.metric("📈 Optimised Predicted Points (XI)", f"{optimised_total_points_xi:.2f}")
                    st.metric("💰 Optimised Team Value (£M)", f"{optimised_total_team_value:.1f}")
                    st.metric("🧢 Optimised Captain", f"{rec_captain_name}")
            # Combine and sort optimised XI and bench by predicted points
            combined_team_df = pd.concat([new_optimised_xi, new_optimised_bench], ignore_index=True)
            combined_team_df = combined_team_df.sort_values(by="pred_points", ascending=False)

            st.markdown("""
                    <h3 style="display: flex; align-items: center; text-align: center;">
                        <span style="flex: 1; height: 3px; background: #ccc;"></span>
                        <span style="padding: 0 10px;">📊 Full Optimised Squad</span>
                        <span style="flex: 1; height: 3px; background: #ccc;"></span>
                    </h3>
                """, unsafe_allow_html=True)
            st.dataframe(
                combined_team_df.reset_index(drop=True)[
                    ['full_name', 'position', 'team_name', 'pred_points', 'value']
                ],
                use_container_width=True,
                hide_index=True
            )

