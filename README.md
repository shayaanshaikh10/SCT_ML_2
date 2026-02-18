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

## License

This project is licensed under the MIT License. See the LICENSE file for details.
