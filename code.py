from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    with open("code.py", "r", encoding="utf-8") as file:
        code_content = file.read()
    return render_template("index.html", code=code_content)

if __name__ == "__main__":
    app.run(debug=True)
