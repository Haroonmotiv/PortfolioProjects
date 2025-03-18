import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import mod  # Import model functions from mod.py

def plot_detailed_analysis(constituency_data, display_name):
    """Display detailed analysis (predicted results) for a given constituency."""
    if not constituency_data.empty:
        # Pie Chart: Vote Share Distribution by Party (Top 6 parties) for predicted results
        vote_share_constituency = constituency_data.groupby("Party")["Predicted_Vote_Share_2026"].first()
        vote_share_constituency = vote_share_constituency.clip(lower=0)
        vote_share_constituency = vote_share_constituency.sort_values(ascending=False).head(6)
        if vote_share_constituency.sum() > 0:
            fig, ax = plt.subplots(figsize=(6,6))
            ax.pie(vote_share_constituency, labels=vote_share_constituency.index,
                   autopct='%1.1f%%', startangle=140)
            ax.set_title("Predicted Vote Share Distribution in " + display_name)
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.write("No valid predicted vote share data available for pie chart in " + display_name)
        
        # Line Chart: Vote Share Trend over Years (2016, 2021, Predicted 2026)
        st.subheader("Predicted Vote Share Trend in " + display_name)
        constituency_summary = constituency_data.set_index("Party")[["Vote_Share_2016", "Vote_Share_2021", "Predicted_Vote_Share_2026"]]
        years = [2016, 2021, 2026]
        fig, ax = plt.subplots(figsize=(8,6))
        for party in constituency_summary.index:
            y_values = constituency_summary.loc[party, ["Vote_Share_2016", "Vote_Share_2021", "Predicted_Vote_Share_2026"]].values
            ax.plot(years, y_values, marker='o', label=party)
        ax.set_xlabel("Year")
        ax.set_ylabel("Vote Share")
        ax.set_title("Predicted Vote Share Trend in " + display_name)
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("No data available for " + display_name)

def plot_historical_analysis(constituency_data, display_name):
    """Display historical analysis for a given constituency using previous election data."""
    # Filter out "tvk" (case-insensitive)
    filtered_data = constituency_data[~constituency_data["Party"].str.lower().eq("tvk")]
    if not filtered_data.empty:
        # Pie Chart: Historical Vote Share Distribution using 2021 data (Top 6 parties)
        hist_vote_share = filtered_data.groupby("Party")["Vote_Share_2021"].first()
        hist_vote_share = hist_vote_share.clip(lower=0)
        hist_vote_share = hist_vote_share.sort_values(ascending=False).head(6)
        if hist_vote_share.sum() > 0:
            fig, ax = plt.subplots(figsize=(6,6))
            ax.pie(hist_vote_share, labels=hist_vote_share.index,
                   autopct='%1.1f%%', startangle=140)
            ax.set_title("Historical Vote Share Distribution (2021) in " + display_name)
            ax.axis('equal')
            st.pyplot(fig)
        else:
            st.write("No valid historical vote share data available for pie chart in " + display_name)
        
        # Line Chart: Historical Vote Share Trend over Years (2016 and 2021)
        st.subheader("Historical Vote Share Trend in " + display_name)
        hist_summary = filtered_data.set_index("Party")[["Vote_Share_2016", "Vote_Share_2021"]]
        years_hist = [2016, 2021]
        fig, ax = plt.subplots(figsize=(8,6))
        for party in hist_summary.index:
            y_values = hist_summary.loc[party, ["Vote_Share_2016", "Vote_Share_2021"]].values
            ax.plot(years_hist, y_values, marker='o', label=party)
        ax.set_xlabel("Year")
        ax.set_ylabel("Vote Share")
        ax.set_title("Historical Vote Share Trend in " + display_name)
        ax.legend()
        st.pyplot(fig)
    else:
        st.write("No historical data available for " + display_name)

def main():
    st.title("Election Prediction Interface")
    st.markdown("""
    This application uses historical election and sentiment data to:
    
    - **Predict 2026 Vote Share** via regression modeling.
    - **Determine Constituency-wise Winners.**
    
    Use the sidebar navigation buttons to display different sections.
    """)
    
    # Run models and retrieve results from mod.py
    with st.spinner("Running models..."):
        results = mod.run_models()
    
    regression = results["regression"]
    merged_data = regression["merged_data"]
    predicted_winners = regression["predicted_winners"]
    win_counts = regression["win_counts"]
    vote_share_percent = regression["vote_share_percent"]
    
    # Check if Constituency_Name exists; if not, try to load and merge Constituency.csv
    if "Constituency_Name" not in merged_data.columns:
        try:
            constituency_map = pd.read_csv("Constituency.csv")
            merged_data = merged_data.merge(constituency_map, on="Constituency_No", how="left")
            if "Constituency_Name" not in predicted_winners.columns:
                predicted_winners = predicted_winners.merge(constituency_map, on="Constituency_No", how="left")
        except Exception as e:
            st.error("Error loading Constituency.csv mapping: " + str(e))
    
    # Sidebar navigation: new order as requested
    section = st.sidebar.radio("Select Section", 
                                ["Predicted Winners", 
                                 "Constituency Win Counts", 
                                 "Detailed Analysis",
                                 "Top 10 Vote Share", 
                                 "Winner Distribution",
                                 "Previous Election Results"])
    
    if section == "Predicted Winners":
        st.header("Predicted Winners for Each Constituency")
        if "Constituency_Name" in predicted_winners.columns:
            cols_to_show = ["Constituency_No", "Constituency_Name", "Predicted_Winner", "Predicted_Vote_Share_2026"]
            existing_cols = [col for col in cols_to_show if col in predicted_winners.columns]
            st.dataframe(predicted_winners[existing_cols])
        else:
            st.dataframe(predicted_winners)
    
    elif section == "Constituency Win Counts":
        st.header("Constituency Win Counts")
        st.dataframe(win_counts)
    
    elif section == "Detailed Analysis":
        # Let the user choose a constituency from the sidebar
        st.sidebar.header("Constituency Details")
        if "Constituency_Name" in merged_data.columns:
            unique_constituencies = merged_data[['Constituency_No', 'Constituency_Name']].drop_duplicates()
            unique_constituencies["Display"] = unique_constituencies["Constituency_Name"] + " (" + unique_constituencies["Constituency_No"].astype(str) + ")"
        else:
            unique_constituencies = merged_data[['Constituency_No']].drop_duplicates()
            unique_constituencies["Display"] = unique_constituencies["Constituency_No"].astype(str)
        options = unique_constituencies.set_index("Display")["Constituency_No"].to_dict()
        selected_display = st.sidebar.selectbox("Select Constituency", list(options.keys()))
        selected_constituency = options[selected_display]
        constituency_data = merged_data[merged_data["Constituency_No"] == selected_constituency]
        st.header("Detailed Analysis for " + selected_display)
        
        # Option to show predicted and historical results in parallel
        show_both = st.checkbox("Show Predicted and Historical Results in Parallel")
        if show_both:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Predicted Results")
                plot_detailed_analysis(constituency_data, selected_display)
            with col2:
                st.markdown("#### Historical Results (TVK excluded)")
                plot_historical_analysis(constituency_data, selected_display)
        else:
            # Default: show only predicted results
            plot_detailed_analysis(constituency_data, selected_display)
            # Option to show historical results below predicted results
            historical_flag = st.checkbox("Show Historical Results for this Constituency")
            if historical_flag:
                st.subheader("Historical Results for " + selected_display)
                plot_historical_analysis(constituency_data, selected_display)
        
        # Option to compare with another constituency
        compare_flag = st.checkbox("Compare with another constituency")
        if compare_flag:
            second_options = {k: v for k, v in options.items() if v != selected_constituency}
            if second_options:
                second_display = st.selectbox("Select second constituency to compare", list(second_options.keys()))
                second_constituency = second_options[second_display]
                second_constituency_data = merged_data[merged_data["Constituency_No"] == second_constituency]
                
                st.subheader("Comparison of Detailed Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"### {selected_display}")
                    plot_detailed_analysis(constituency_data, selected_display)
                    if st.checkbox("Show Historical Results for " + selected_display, key="hist1"):
                        plot_historical_analysis(constituency_data, selected_display)
                with col2:
                    st.markdown(f"### {second_display}")
                    plot_detailed_analysis(second_constituency_data, second_display)
                    if st.checkbox("Show Historical Results for " + second_display, key="hist2"):
                        plot_historical_analysis(second_constituency_data, second_display)
            else:
                st.write("No other constituency available for comparison.")
    
    elif section == "Top 10 Vote Share":
        st.header("Predicted Vote Share for 2026 (Top 10 Parties)")
        vote_share_df = vote_share_percent.reset_index().rename(columns={'index': 'Party', 0: 'Vote Share (%)'})
        st.dataframe(vote_share_df)
        
        fig2, ax2 = plt.subplots(figsize=(6,6))
        ax2.pie(vote_share_percent, labels=vote_share_percent.index,
                autopct='%1.1f%%', startangle=140)
        ax2.set_title("Predicted Vote Share (Top 10 Parties)")
        ax2.axis('equal')
        st.pyplot(fig2)
    
    elif section == "Winner Distribution":
        st.header("Constituency Winner Distribution")
        if not win_counts.empty:
            fig1, ax1 = plt.subplots(figsize=(6,6))
            ax1.pie(win_counts["Constituencies_Won"], labels=win_counts["Party"],
                    autopct='%1.1f%%', startangle=140)
            ax1.set_title("Constituency Winner Distribution")
            ax1.axis('equal')
            st.pyplot(fig1)
        else:
            st.write("No win counts available for pie chart.")
    
    elif section == "Previous Election Results":
        st.header("Previous Election Results")
        year_choice = st.radio("Select Year", ["2016 result", "2021 result"])
        if year_choice == "2016 result":
            st.subheader("2016 Election Results")
            df_2016 = merged_data.groupby("Party")["Vote_Share_2016"].mean().reset_index()
            st.dataframe(df_2016)
            top10_2016 = df_2016.sort_values("Vote_Share_2016", ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(6,6))
            ax.pie(top10_2016["Vote_Share_2016"], labels=top10_2016["Party"],
                   autopct='%1.1f%%', startangle=140)
            ax.set_title("2016 Vote Share Distribution (Top 10 Parties)")
            ax.axis('equal')
            st.pyplot(fig)
            fig2, ax2 = plt.subplots(figsize=(8,6))
            ax2.bar(top10_2016["Party"], top10_2016["Vote_Share_2016"])
            ax2.set_xlabel("Party")
            ax2.set_ylabel("Average Vote Share")
            ax2.set_title("Average Vote Share in 2016 by Party")
            st.pyplot(fig2)
        elif year_choice == "2021 result":
            st.subheader("2021 Election Results")
            df_2021 = merged_data.groupby("Party")["Vote_Share_2021"].mean().reset_index()
            st.dataframe(df_2021)
            top10_2021 = df_2021.sort_values("Vote_Share_2021", ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(6,6))
            ax.pie(top10_2021["Vote_Share_2021"], labels=top10_2021["Party"],
                   autopct='%1.1f%%', startangle=140)
            ax.set_title("2021 Vote Share Distribution (Top 10 Parties)")
            ax.axis('equal')
            st.pyplot(fig)
            fig2, ax2 = plt.subplots(figsize=(8,6))
            ax2.bar(top10_2021["Party"], top10_2021["Vote_Share_2021"])
            ax2.set_xlabel("Party")
            ax2.set_ylabel("Average Vote Share")
            ax2.set_title("Average Vote Share in 2021 by Party")
            st.pyplot(fig2)

if __name__ == "__main__":
    main()
