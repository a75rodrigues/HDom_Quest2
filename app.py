import io
from flask import Flask, request, send_file, render_template, redirect, url_for, flash
from processor import process_uploaded_files

app = Flask(__name__)
app.secret_key = "questionario-secret-key"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    model_file = request.files.get("modelo")
    pdf_files = request.files.getlist("pdfs")

    if model_file is None or model_file.filename == "":
        flash("Escolhe primeiro o modelo em branco.")
        return redirect(url_for("index"))

    pdf_files = [f for f in pdf_files if f and f.filename]
    if not pdf_files:
        flash("Escolhe pelo menos um PDF preenchido.")
        return redirect(url_for("index"))

    try:
        result_bytes = process_uploaded_files(
            model_file=model_file,
            pdf_files=pdf_files,
            config_path="config.json"
        )
    except Exception as e:
        flash(f"Erro no processamento: {e}")
        return redirect(url_for("index"))

    return send_file(
        io.BytesIO(result_bytes),
        as_attachment=True,
        download_name="resultado.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
