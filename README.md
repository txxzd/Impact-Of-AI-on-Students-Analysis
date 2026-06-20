# Impact of AI on Students - Exploratory Data Analysis and Dashboard

An exploratory data analysis (EDA) and Streamlit dashboard for the [AI Impact on Students](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students/data) dataset from Kaggle, looking at how generative AI usage relates to student academic performance, study habits, and well-being.

## Dataset

- **Source:** [AI Impact on Students (Kaggle)](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students/data)
- **File:** `data.csv`
- **Size:** 50,000 rows x 16 columns

### Columns

| Column | Description |
| --- | --- |
| `Student_ID` | Unique identifier for each student |
| `Major_Category` | Field of study (e.g., STEM, Business, Arts, Humanities, Medical) |
| `Year_of_Study` | Academic year (Freshman, Sophomore, Junior, Senior, Graduate) |
| `Pre_Semester_GPA` | GPA before the semester |
| `Weekly_GenAI_Hours` | Hours per week spent using generative AI tools |
| `Primary_Use_Case` | Main reason for using GenAI (e.g., Ideation, Copywriting/Drafting, Debugging/Troubleshooting, Summarizing/Reading) |
| `Prompt_Engineering_Skill` | Self-rated skill level (Beginner, Intermediate, Advanced) |
| `Tool_Diversity` | Number of distinct AI tools used |
| `Paid_Subscription` | Whether the student pays for an AI tool subscription |
| `Traditional_Study_Hours` | Hours per week spent on traditional (non-AI) studying |
| `Perceived_AI_Dependency` | Self-rated dependency on AI tools (scale) |
| `Institutional_Policy` | School's policy on AI use (Strict_Ban, Allowed_With_Citation, Actively_Encouraged) |
| `Anxiety_Level_During_Exams` | Self-rated exam anxiety (scale) |
| `Post_Semester_GPA` | GPA after the semester |
| `Skill_Retention_Score` | Score representing retention of learned skills |
| `Burnout_Risk_Level` | Burnout risk category (Low, Medium, High) |

## Dashboard Features

- Sidebar filters for major, year, policy, paid access, prompt skill, and burnout level
- Overview metrics for usage, GPA change, retention, and improvement rate
- Usage visualizations for majors, use cases, skill distribution, and policy differences
- Outcomes and well-being charts for retention, burnout, dependency, anxiety, and a Spearman correlation heatmap
- Policy summary with GPA comparisons and statistical test output

## Project Structure

```text
.
|- data.csv
|- data_analysis.ipynb
|- app.py
|- requirements.txt
`- README.md
```

## Getting Started

### Requirements

- Python 3.9+
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter notebook data_analysis.ipynb
```

### Running the Dashboard

```bash
streamlit run app.py
```

## Interpretation Note

This project is exploratory and observational. The charts and statistical tests in the notebook and dashboard should be interpreted as showing associations and group differences, not proof of causation.

## License

The dataset is provided by [laveshjadon on Kaggle](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students/data); refer to the Kaggle page for licensing and usage terms.
