import os
import PyPDF2
import pickle
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug.utils import secure_filename
UPLOAD_FOLDER='./uploads/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
clf = pickle.load(open('model.pkl','rb'))
loaded_vec = pickle.load(open("fitted_vectorizer.pkl", "rb"))

@app.route('/')
def predict():
    return render_template('category_pred.html')

@app.route('/result',methods=['GET','POST'])
def upload_files():
    if request.method == 'POST':
        file = request.files['filename']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        pdf_file_obj = open('./uploads/'+filename,'rb')
        pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)
        #page_obj = pdf_reader.getPage(0)
        count = pdf_reader.numPages
        for i in range(count):
            page =pdf_reader.getPage(i)
            result=page.extractText()
        #result = page_obj.extractText()
        result_pred = clf.predict(loaded_vec.transform([result]))
        return render_template('category_result.html', result=result_pred)


if __name__=="__main__":
    app.debug=True
    app.run()
