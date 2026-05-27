from flask import Flask,render_template,request
import os
from predict import predict_disease
from disease_info import disease_info

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/",methods=["GET","POST"])
def home():

    result=None

    if request.method=="POST":

        file = request.files["image"]

        path = os.path.join(UPLOAD_FOLDER,file.filename)

        file.save(path)

        disease,confidence = predict_disease(path)

        info = disease_info.get(disease,{
            "description":"Information not available",
            "remedy":"Consult agricultural expert"
        })

        result={
            "image":path,
            "disease":disease,
            "confidence":round(confidence,2),
            "description":info["description"],
            "remedy":info["remedy"]
        }

    return render_template("index.html",result=result)


if __name__=="__main__":
    app.run(debug=True)
