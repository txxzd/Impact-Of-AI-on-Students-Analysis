from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


st.set_page_config(
    page_title="Impact of AI on Students Dashboard",
    page_icon="📊",
    layout="wide",
)

sns.set_theme(style="whitegrid")

DATA_PATH = Path(__file__).with_name("data.csv")
YEAR_ORDER = ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]
SKILL_ORDER = ["Beginner", "Intermediate", "Advanced"]
BURNOUT_ORDER = ["Low", "Medium", "High"]
POLICY_ORDER = ["Strict_Ban", "Allowed_With_Citation", "Actively_Encouraged"]
ACCENT = "#0f766e"
ACCENT_LIGHT = "#99f6e4"
SLATE = "#475569"
CARD_BG = "#f8fafc"
CARD_BORDER = "#e2e8f0"
GOOD = "#15803d"
WARN = "#b45309"


st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 24%, #f8fafc 100%);
    }}
    .hero {{
        background: linear-gradient(135deg, #0f172a 0%, #134e4a 52%, #0f766e 100%);
        border-radius: 8px;
        padding: 1.6rem 1.6rem 1.3rem 1.6rem;
        color: #f8fafc;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.16);
    }}
    .hero h1 {{
        margin: 0 0 0.45rem 0;
        font-size: 2rem;
        line-height: 1.05;
    }}
    .hero p {{
        margin: 0;
        max-width: 52rem;
        color: rgba(248,250,252,0.88);
        font-size: 0.96rem;
    }}
    .metric-card {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 8px;
        padding: 0.9rem 1rem;
        min-height: 96px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }}
    .metric-label {{
        color: {SLATE};
        font-size: 0.82rem;
        margin-bottom: 0.35rem;
    }}
    .metric-value {{
        color: #0f172a;
        font-size: 1.7rem;
        font-weight: 700;
        line-height: 1.1;
    }}
    .metric-note {{
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 0.45rem;
    }}
    .section-note {{
        background: #ffffff;
        border: 1px solid {CARD_BORDER};
        border-left: 4px solid {ACCENT};
        border-radius: 8px;
        padding: 0.8rem 0.95rem;
        margin: 0.15rem 0 1rem 0;
        color: #334155;
        font-size: 0.92rem;
    }}
    .insight-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 0.4rem;
    }}
    .insight-card {{
        background: #ffffff;
        border: 1px solid {CARD_BORDER};
        border-radius: 8px;
        padding: 0.9rem 1rem;
    }}
    .insight-title {{
        color: #0f172a;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.35rem;
    }}
    .insight-body {{
        color: #475569;
        font-size: 0.86rem;
        line-height: 1.45;
    }}
    .summary-table {{
        width: 100%;
        border-collapse: collapse;
        background: #ffffff;
        border: 1px solid {CARD_BORDER};
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }}
    .summary-table th {{
        text-align: left;
        background: #f8fafc;
        color: #334155;
        font-size: 0.82rem;
        padding: 0.8rem 0.9rem;
        border-bottom: 1px solid {CARD_BORDER};
    }}
    .summary-table td {{
        padding: 0.78rem 0.9rem;
        border-bottom: 1px solid #eef2f7;
        font-size: 0.88rem;
        color: #0f172a;
    }}
    .summary-table tr:last-child td {{
        border-bottom: none;
    }}
    .summary-table td:last-child {{
        text-align: right;
        font-weight: 600;
        color: {SLATE};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.72);
        padding: 0.35rem 0.25rem 0 0.25rem;
        border-bottom: 1px solid #e5e7eb;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: #475569;
        font-weight: 600;
        padding: 0.65rem 0.8rem 0.7rem 0.8rem;
    }}
    .stTabs [aria-selected="true"] {{
        color: #ef4444 !important;
    }}
    .stTabs [aria-selected="false"] {{
        color: #64748b !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    df["GPA_Change"] = df["Post_Semester_GPA"] - df["Pre_Semester_GPA"]
    df["Year_of_Study"] = pd.Categorical(df["Year_of_Study"], categories=YEAR_ORDER, ordered=True)
    df["Prompt_Engineering_Skill"] = pd.Categorical(
        df["Prompt_Engineering_Skill"], categories=SKILL_ORDER, ordered=True
    )
    df["Burnout_Risk_Level"] = pd.Categorical(
        df["Burnout_Risk_Level"], categories=BURNOUT_ORDER, ordered=True
    )
    df["Institutional_Policy"] = pd.Categorical(
        df["Institutional_Policy"], categories=POLICY_ORDER, ordered=True
    )
    df["GenAI_Usage_Group"] = pd.cut(
        df["Weekly_GenAI_Hours"],
        bins=[0, 5, 10, 20, 40],
        labels=["Low (0-5)", "Medium (5-10)", "High (10-20)", "Very High (20+)"],
        include_lowest=True,
    )
    return df


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


def safe_corr(series_a: pd.Series, series_b: pd.Series, method: str = "pearson") -> float:
    corr = series_a.corr(series_b, method=method)
    return 0.0 if pd.isna(corr) else float(corr)


def build_spearman_heatmap_df(df: pd.DataFrame) -> pd.DataFrame:
    heatmap_df = df.copy()
    heatmap_df["Year_of_Study_Num"] = heatmap_df["Year_of_Study"].map(
        {
            "Freshman": 1,
            "Sophomore": 2,
            "Junior": 3,
            "Senior": 4,
            "Graduate": 5,
        }
    )
    heatmap_df["Prompt_Engineering_Skill_Num"] = heatmap_df["Prompt_Engineering_Skill"].map(
        {
            "Beginner": 1,
            "Intermediate": 2,
            "Advanced": 3,
        }
    )
    heatmap_df["Burnout_Risk_Level_Num"] = heatmap_df["Burnout_Risk_Level"].map(
        {
            "Low": 1,
            "Medium": 2,
            "High": 3,
        }
    )
    heatmap_df["Paid_Subscription_Num"] = heatmap_df["Paid_Subscription"].astype(int)

    heatmap_cols = [
        "Year_of_Study_Num",
        "Pre_Semester_GPA",
        "Post_Semester_GPA",
        "GPA_Change",
        "Weekly_GenAI_Hours",
        "Prompt_Engineering_Skill_Num",
        "Tool_Diversity",
        "Paid_Subscription_Num",
        "Traditional_Study_Hours",
        "Perceived_AI_Dependency",
        "Anxiety_Level_During_Exams",
        "Skill_Retention_Score",
        "Burnout_Risk_Level_Num",
    ]

    return heatmap_df[heatmap_cols].rename(
        columns={
            "Year_of_Study_Num": "Year_of_Study",
            "Prompt_Engineering_Skill_Num": "Prompt_Engineering_Skill",
            "Paid_Subscription_Num": "Paid_Subscription",
            "Burnout_Risk_Level_Num": "Burnout_Risk_Level",
        }
    )


def build_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    major_options = sorted(df["Major_Category"].dropna().unique().tolist())
    major = st.sidebar.multiselect("Major Category", options=major_options, default=major_options)
    year = st.sidebar.multiselect("Year of Study", options=YEAR_ORDER, default=YEAR_ORDER)
    policy = st.sidebar.multiselect("Institutional Policy", options=POLICY_ORDER, default=POLICY_ORDER)
    paid = st.sidebar.multiselect(
        "Paid Subscription",
        options=[False, True],
        default=[False, True],
        format_func=lambda value: "Paid" if value else "Free",
    )
    skill = st.sidebar.multiselect("Prompt Engineering Skill", options=SKILL_ORDER, default=SKILL_ORDER)
    burnout = st.sidebar.multiselect("Burnout Risk Level", options=BURNOUT_ORDER, default=BURNOUT_ORDER)

    filtered = df[
        df["Major_Category"].isin(major)
        & df["Year_of_Study"].isin(year)
        & df["Institutional_Policy"].isin(policy)
        & df["Paid_Subscription"].isin(paid)
        & df["Prompt_Engineering_Skill"].isin(skill)
        & df["Burnout_Risk_Level"].isin(burnout)
    ].copy()
    return filtered


def metric_card(label: str, value: str, note: str) -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div>'
        f"</div>"
    )


def render_summary_table(df: pd.DataFrame) -> None:
    rows = "".join(
        f"<tr><td>{metric}</td><td>{value}</td></tr>"
        for metric, value in zip(df["Metric"], df["Value"])
    )
    st.markdown(
        f"""
        <table class="summary-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(df: pd.DataFrame) -> None:
    improved_pct = (df["GPA_Change"] > 0).mean() * 100 if len(df) else 0
    high_burnout_pct = (df["Burnout_Risk_Level"] == "High").mean() * 100 if len(df) else 0
    dep_anx_corr = safe_corr(
        df["Perceived_AI_Dependency"], df["Anxiety_Level_During_Exams"], method="spearman"
    )
    usage_gpa_corr = safe_corr(df["Weekly_GenAI_Hours"], df["GPA_Change"])

    cols = st.columns(5)
    cards = [
        metric_card("Students", f"{len(df):,}", "Rows in the current filtered view"),
        metric_card("Avg Weekly GenAI Hours", f"{df['Weekly_GenAI_Hours'].mean():.2f}", "Typical weekly usage intensity"),
        metric_card("Avg GPA Change", f"{df['GPA_Change'].mean():.3f}", "Post semester GPA minus pre semester GPA"),
        metric_card("Avg Skill Retention", f"{df['Skill_Retention_Score'].mean():.1f}", "Average retained learning score"),
        metric_card("Improved GPA", format_pct(improved_pct), "Share of students with positive GPA change"),
    ]
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="section-note">
            High burnout share in the current view: <strong>{format_pct(high_burnout_pct)}</strong> |
            Dependency-anxiety Spearman correlation: <strong>{dep_anx_corr:.2f}</strong> |
            Usage-GPA correlation: <strong>{usage_gpa_corr:.2f}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_tab(df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-note">A quick scan of the filtered population: usage intensity, GPA movement, burnout mix, and whether heavier use clusters with better academic change.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df["Weekly_GenAI_Hours"], bins=30, kde=True, ax=ax, color=ACCENT)
        ax.set_title("Weekly GenAI Hours Distribution")
        ax.set_xlabel("Hours per Week")
        st.pyplot(fig, clear_figure=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df["GPA_Change"], bins=30, kde=True, ax=ax, color="#22c55e")
        ax.set_title("GPA Change Distribution")
        ax.set_xlabel("Post Semester GPA - Pre Semester GPA")
        st.pyplot(fig, clear_figure=True)

    with right:
        burnout_counts = (
            df["Burnout_Risk_Level"].value_counts(normalize=True).reindex(BURNOUT_ORDER).fillna(0) * 100
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(
            x=burnout_counts.index,
            y=burnout_counts.values,
            ax=ax,
            palette=["#a7f3d0", "#fbbf24", "#ef4444"],
        )
        ax.set_title("Burnout Risk Distribution")
        ax.set_ylabel("Percent of Students")
        ax.set_xlabel("")
        st.pyplot(fig, clear_figure=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(
            data=df,
            x="Weekly_GenAI_Hours",
            y="GPA_Change",
            alpha=0.2,
            s=22,
            ax=ax,
            color="#3b82f6",
        )
        ax.axhline(0, color="#ef4444", linestyle="--", linewidth=1)
        ax.set_title("GPA Change vs Weekly GenAI Hours")
        ax.set_xlabel("Weekly GenAI Hours")
        ax.set_ylabel("GPA Change")
        st.pyplot(fig, clear_figure=True)

    summary = pd.DataFrame(
        {
            "Metric": [
                "Average Tool Diversity",
                "Average Traditional Study Hours",
                "Average AI Dependency",
                "Average Exam Anxiety",
                "Correlation: GenAI Hours vs GPA Change",
            ],
            "Value": [
                round(df["Tool_Diversity"].mean(), 2),
                round(df["Traditional_Study_Hours"].mean(), 2),
                round(df["Perceived_AI_Dependency"].mean(), 2),
                round(df["Anxiety_Level_During_Exams"].mean(), 2),
                round(safe_corr(df["Weekly_GenAI_Hours"], df["GPA_Change"]), 3),
            ],
        }
    )
    st.subheader("Current View Summary")
    render_summary_table(summary)


def render_usage_tab(df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-note">This section focuses on who uses GenAI, how they use it, and whether behavior differs more by academic profile, institutional policy, or paid access.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)

    with left:
        cross_tab_pct = pd.crosstab(df["Major_Category"], df["Primary_Use_Case"], normalize="index") * 100
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cross_tab_pct, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
        ax.set_title("Primary Use Case by Major (%)")
        st.pyplot(fig, clear_figure=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(
            data=df,
            x="Institutional_Policy",
            y="Weekly_GenAI_Hours",
            order=POLICY_ORDER,
            ax=ax,
            palette=["#34d399", "#60a5fa", "#f59e0b"],
        )
        ax.set_title("Weekly GenAI Hours by Institutional Policy")
        ax.set_xlabel("")
        ax.set_ylabel("Weekly GenAI Hours")
        st.pyplot(fig, clear_figure=True)

    with right:
        skill_by_year = pd.crosstab(df["Year_of_Study"], df["Prompt_Engineering_Skill"], normalize="index") * 100
        skill_by_year = skill_by_year.reindex(YEAR_ORDER).fillna(0)
        fig, ax = plt.subplots(figsize=(7, 5))
        skill_by_year.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=["#d9d9d9", "#38bdf8", "#0f766e"],
        )
        ax.set_title("Prompt Engineering Skill by Year of Study (%)")
        ax.set_xlabel("")
        ax.set_ylabel("Percentage (%)")
        ax.legend(title="Skill", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.xticks(rotation=0)
        st.pyplot(fig, clear_figure=True)

        paid_summary = (
            df.groupby("Paid_Subscription")[["Weekly_GenAI_Hours", "Tool_Diversity"]]
            .median()
            .rename(index={False: "Free", True: "Paid"})
        )
        st.subheader("Paid vs Free")
        st.dataframe(paid_summary, use_container_width=True)

    st.markdown(
        """
        <div class="insight-grid">
            <div class="insight-card">
                <div class="insight-title">What to look for</div>
                <div class="insight-body">Compare whether usage differences are driven more by majors and tasks, or by access conditions like paid subscriptions and school policy.</div>
            </div>
            <div class="insight-card">
                <div class="insight-title">Reading cue</div>
                <div class="insight-body">If the major-use-case heatmap shows sharper contrast than the policy boxplot, the story is more task-driven than rule-driven.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_outcomes_tab(df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-note">The outcomes view shifts from behavior to impact: GPA movement, retained learning, burnout, dependency, anxiety, and how these pieces move together.</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(
            data=df,
            x="Prompt_Engineering_Skill",
            y="Skill_Retention_Score",
            order=SKILL_ORDER,
            ax=ax,
            palette=["#d9d9d9", "#38bdf8", "#0f766e"],
        )
        ax.set_title("Skill Retention by Prompt Engineering Skill")
        ax.set_xlabel("")
        ax.set_ylabel("Skill Retention Score")
        st.pyplot(fig, clear_figure=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(
            data=df,
            x="Burnout_Risk_Level",
            y="Perceived_AI_Dependency",
            order=BURNOUT_ORDER,
            ax=ax,
            palette=["#a7f3d0", "#fbbf24", "#ef4444"],
        )
        ax.set_title("AI Dependency by Burnout Risk Level")
        ax.set_xlabel("")
        ax.set_ylabel("Perceived AI Dependency")
        st.pyplot(fig, clear_figure=True)

    with right:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(
            data=df,
            x="Burnout_Risk_Level",
            y="Weekly_GenAI_Hours",
            order=BURNOUT_ORDER,
            ax=ax,
            palette=["#a7f3d0", "#fbbf24", "#ef4444"],
        )
        ax.set_title("Weekly GenAI Hours by Burnout Risk Level")
        ax.set_xlabel("")
        ax.set_ylabel("Weekly GenAI Hours")
        st.pyplot(fig, clear_figure=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(
            data=df,
            x="Perceived_AI_Dependency",
            y="Anxiety_Level_During_Exams",
            ax=ax,
            palette="Purples",
        )
        ax.set_title("Exam Anxiety by AI Dependency")
        ax.set_xlabel("Perceived AI Dependency")
        ax.set_ylabel("Anxiety Level During Exams")
        st.pyplot(fig, clear_figure=True)

    retention_means = (
        df.groupby("Prompt_Engineering_Skill")["Skill_Retention_Score"].mean().reindex(SKILL_ORDER).round(2)
    )
    burnout_means = df.groupby("Burnout_Risk_Level")["Weekly_GenAI_Hours"].mean().reindex(BURNOUT_ORDER).round(2)

    st.write(
        f"Prompt engineering skill shows a clearer association with retention {retention_means.to_dict()} than "
        f"weekly GenAI hours do, while average GenAI hours also rise across burnout levels {burnout_means.to_dict()}."
    )

    st.subheader("Spearman Correlation Heatmap")
    spearman_df = build_spearman_heatmap_df(df)
    corr_matrix = spearman_df.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Spearman Correlation Heatmap")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    st.pyplot(fig, clear_figure=True)

    st.markdown(
        """
        <div class="insight-grid">
            <div class="insight-card">
                <div class="insight-title">Why this view matters</div>
                <div class="insight-body">The heatmap compresses the main outcomes story into one matrix, making it easier to compare GPA change, retention, burnout, dependency, and anxiety side by side.</div>
            </div>
            <div class="insight-card">
                <div class="insight-title">Interpretation note</div>
                <div class="insight-body">These are ranked associations in the filtered sample, not causal effects. Stronger colors mean stronger monotonic relationships, not proof of mechanism.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_policy_tab(df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="section-note">This section isolates the policy question: whether GPA change differs across institutional rules and whether that difference survives post-hoc testing.</div>',
        unsafe_allow_html=True,
    )
    group_means = df.groupby("Institutional_Policy")["GPA_Change"].mean().reindex(POLICY_ORDER)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        x=group_means.index,
        y=group_means.values,
        ax=ax,
        palette=["#34d399", "#60a5fa", "#f59e0b"],
    )
    ax.set_title("Average GPA Change by Institutional Policy")
    ax.set_xlabel("")
    ax.set_ylabel("Average GPA Change")
    ax.tick_params(axis="x", rotation=10)
    st.pyplot(fig, clear_figure=True)

    policy_df = df.dropna(subset=["Institutional_Policy", "GPA_Change"]).copy()
    available_policies = [policy for policy in POLICY_ORDER if policy_df["Institutional_Policy"].eq(policy).any()]

    st.subheader("Statistical Summary")
    if len(available_policies) >= 2:
        groups = [policy_df.loc[policy_df["Institutional_Policy"] == policy, "GPA_Change"] for policy in available_policies]

        if len(available_policies) >= 3:
            f_stat, p_value = stats.f_oneway(*groups)
            st.write(f"ANOVA on GPA change across policy groups: F = {f_stat:.3f}, p = {p_value:.4g}")
        else:
            t_stat, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            st.write(f"Two-group comparison on GPA change: t = {t_stat:.3f}, p = {p_value:.4g}")

        if len(available_policies) >= 3:
            tukey = pairwise_tukeyhsd(
                endog=policy_df["GPA_Change"],
                groups=policy_df["Institutional_Policy"],
                alpha=0.05,
            )
            tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
            st.dataframe(tukey_df, use_container_width=True, hide_index=True)

        st.caption(
            "These comparisons are observational. They show differences in group averages, not proof that policy causes the change."
        )
    else:
        st.info("Select at least two policy groups to compare GPA change.")


def main() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Impact of AI on Students</h1>
            <p>
                An interactive portfolio dashboard built from the notebook analysis. Use the filters to compare how
                GenAI usage, study habits, retention, burnout, and exam anxiety shift across student groups.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()
    filtered = build_filters(df)

    if filtered.empty:
        st.warning("No rows match the current filters. Expand the selection to continue.")
        return

    render_metric_row(filtered)

    overview_tab, usage_tab, outcomes_tab, policy_tab = st.tabs(
        ["Overview", "Usage Patterns", "Outcomes & Wellbeing", "Policy"]
    )

    with overview_tab:
        render_overview_tab(filtered)
    with usage_tab:
        render_usage_tab(filtered)
    with outcomes_tab:
        render_outcomes_tab(filtered)
    with policy_tab:
        render_policy_tab(filtered)


if __name__ == "__main__":
    main()
