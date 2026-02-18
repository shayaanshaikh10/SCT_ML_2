# Customer Segmentation with K-means

This project segments retail customers using the **Mall Customers dataset** from Kaggle:

- Dataset URL: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

## Project Objective

Group customers into clusters based on purchase-related behavior using **K-means clustering**.

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Kneed (automatic elbow detection)

## Project Structure

```text
SCT_ML_2/
├── data/
│   └── Mall_Customers.csv         # Add dataset here
├── outputs/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   └── segment_customers.py
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

1. Download the dataset from Kaggle.
2. Put `Mall_Customers.csv` inside the `data/` folder.
3. Run:

```bash
python src/segment_customers.py --input data/Mall_Customers.csv --output-dir outputs --max-k 10
```

## Dataset Policy

- Do not commit Kaggle dataset files to GitHub.
- Keep only `data/.gitkeep` in the repository and download `Mall_Customers.csv` locally when running the project.

## Outputs

After running, the following files are generated inside `outputs/`:

- `clustered_customers.csv` - Original data + assigned cluster label
- `cluster_summary.csv` - Per-cluster average stats and customer count
- `silhouette_scores.csv` - Silhouette score for each tested K (2 to max-k)
- `elbow_plot.png` - Elbow chart for K selection
- `cluster_plot.png` - Segment visualization (income vs spending score)

## Notes for Internship Submission

- Explain why K-means was selected (simple, fast, interpretable baseline).
- Mention feature scaling and why it matters.
- Include the elbow and cluster plots in your report/presentation.
- Add business interpretation for each cluster (e.g., high income/high spend).

## License

This project is licensed under the MIT License. See the LICENSE file for details.
