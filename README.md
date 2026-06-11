# Impact of AI on Students — Exploratory Data Analysis

An exploratory data analysis (EDA) of the [AI Impact on Students](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students/data) dataset from Kaggle, looking at how generative AI usage relates to student academic performance, study habits, and well-being.

## Dataset

- **Source:** [AI Impact on Students (Kaggle)](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students/data)
- **File:** `data.csv`
- **Size:** 50,000 rows × 16 columns

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

## Goals of the Analysis

- Understand the distribution of GenAI usage across majors, years, and institutional policies
- Examine the relationship between GenAI usage and the change in GPA (`Post_Semester_GPA` vs `Pre_Semester_GPA`)
- Explore how AI dependency, prompt engineering skill, and tool diversity relate to skill retention
- Investigate links between AI usage patterns, exam anxiety, and burnout risk

## Project Structure

```
.
├── data.csv              # Raw dataset (50,000 students, 16 features)
├── data_analysis.ipynb   # Jupyter notebook containing the EDA
└── README.md
```

## Getting Started

### Requirements

- Python 3.9+
- pandas
- matplotlib / seaborn
- jupyter

Install dependencies:

```bash
pip install pandas matplotlib seaborn jupyter
```

### Running the Notebook

```bash
jupyter notebook data_analysis.ipynb
```

## License

The dataset is provided by [laveshjadon on Kaggle](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students/data); refer to the Kaggle page for licensing and usage terms.
