Telecom Customer Segmentation

An educational Python project that explores telecom customer data, creates customer groups using K-means clustering,
and displays the results in an interactive Streamlit dashboard.

Project files

python.py — cleans the data, explores churn, prepares features, creates customer segments, and saves charts and results.

dashboard.py — displays interactive charts, filters, customer records, and segment summaries.

Dataset

This project uses the public Telco Customer Churn dataset. It is sample data, not confidential data from an employer.
Download the CSV and make sure the file path near the top of python.py points to where you saved it. The script currently expects:
C:\ALGONIVE\archive\WA_Fn-UseC_-Telco-Customer-Churn.csv
Setup and run

Install the required packages:

python -m pip install pandas matplotlib scikit-learn plotly streamlit

Open a terminal in the folder containing python.py and dashboard.py. Run the analysis first:

python python.py

Then start the dashboard from that same folder:

streamlit run dashboard.py

The analysis creates telco_customers_with_segments.csv, which the dashboard needs to load.

How the segmentation works

The project prepares numeric and service-related customer features, scales numeric values, and converts categories into numeric 
indicators. It compares K-means cluster counts using silhouette scores and assigns each customer a segment.

customerID stays in the results so a customer’s row can be identified, but it is excluded from clustering because 
it is only an identifier. Churn is used to describe the resulting segments, not as a clustering input.
