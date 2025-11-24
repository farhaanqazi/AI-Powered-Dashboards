from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    return render_template ("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    # Get the uploaded file from the form
    uploaded_file = request.files.get("dataset")

    if not uploaded_file:
        return "No file uploaded", 400

    # For now, just show the file name as a simple confirmation
    filename = uploaded_file.filename
    return f"File '{filename}' uploaded successfully! (No ML yet)"


if __name__ == "__main__":
    app.run(debug=True)
