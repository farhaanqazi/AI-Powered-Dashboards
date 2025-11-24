from flask import Flask, render_template, request
from src.data.parser import load_csv
from src.data.analyser import basic_profile

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    uploaded_file = request.files.get("dataset")

    if not uploaded_file:
        return "No file uploaded", 400

    # Load the CSV into a pandas DataFrame
    df = load_csv(uploaded_file)

    if df is None:
        return "Failed to read CSV file", 400

    # For now, just print the first 5 rows to the terminal
    print("Preview of uploaded data:")
    print(df.head())
    
    # Basic profile
    profile = basic_profile(df)
    print("\nBasic profile:")
    for col_info in profile:
        print(col_info)

    # Show the profile in an HTML table
    return render_template("dashboard.html", profile=profile)

if __name__ == "__main__":
    app.run(debug=True)
