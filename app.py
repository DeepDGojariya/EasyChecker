from flask import Flask,render_template,request
import numpy as np
import tensorflow 
from keras.preprocessing import image
from keras.models import load_model
import pickle
from PIL import Image

app = Flask(__name__)


def predict_heart(age,sex,rbp,chol,fbs,cpt,ecg,maxhr,exang,oldpeak,slope,fcolor,thal):
    model = pickle.load(open('heart_disease_model.sav','rb'))
    prediction = model.predict_proba([[int(age),int(sex),int(cpt),int(rbp),int(chol),float(fbs),int(ecg),int(maxhr),int(exang),float(oldpeak),int(slope),int(fcolor),int(thal)]])
    if prediction[0][0] > prediction[0][1]:
        return 'Low Risk'
    else:
        return 'High Risk'

def predict_diabetes(preg,glu,bp,st,ins,bmi,dbf,age):
    model = pickle.load(open('diabetes_model.sav','rb'))
    prediction = model.predict_proba([[int(preg),int(glu),int(bp),int(st),int(ins),float(bmi),float(dbf),int(age)]])
    if prediction[0][0] > prediction[0][1]:
        return 'Low Risk'
    else:
        return 'High Risk'

'''
def predict_class(img_path):
    model = load_model('pneumonia_tf1.h5')
    test_image = image.load_img(img_path, target_size=(256,256))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    result=list(result[0])
    return result[0]

'''


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/heartdisease',methods=['GET','POST'])
def heart_disease():
    if request.method == 'GET':
        return render_template('heartdisease.html')
    else:
        age = request.form['age']
        sex = request.form['sex']
        rbp = request.form['rbp']
        chol = request.form['chol']
        fbs = request.form['fbs']
        cpt = request.form['cpt']
        ecg = request.form['ecg']
        maxhr = request.form['max']
        exang = request.form['exang']
        oldpeak = request.form['old']
        slope = request.form['slope']
        fcolor = request.form['color']
        thal = request.form['thal']
        output = predict_heart(age,sex,rbp,chol,fbs,cpt,ecg,maxhr,exang,oldpeak,slope,fcolor,thal)
        if sex=="0":
            sex="Female"
        else:
            sex="Male"

        if cpt=="0":
            cpt="No-Pain"
        elif cpt=="1":
            cpt="Little Pain"
        elif cpt=="2":
            cpt="Pain On Hit"
        else:
            cpt="Very High Pain"
        
        if ecg=="0":
            ecg="Normal"
        elif ecg=="1":
            ecg="Having ST-T Pain"
        else:
            ecg="Hypertropy"
        
        if exang=="0":
            exang="No"
        else:
            exang="Yes"
        
        if slope=="0":
            slope="Upsloping"
        elif slope=="1":
            slope="Flat"
        else:
            slope="Downsloping"
        
        if thal=="3":
            thal="Normal"
        elif thal=="6":
            thal="Fixed Defect"
        else:
            thal="Reversible Defect"
        
        context = {'output':output,'age':age,'sex':sex,'rbp':rbp,'chol':chol,'fbs':fbs,'cpt':cpt,'ecg':ecg,'maxhr':maxhr,'exang':exang,'oldpeak':oldpeak,
                    'slope':slope,'fcolor':fcolor,'thal':thal,'heart':1}

        return render_template('results.html',context = context)


@app.route('/diabetes',methods=['GET','POST'])
def diabetes():
    if request.method == 'GET':
        return render_template('diabetes.html')
    else:
        age = request.form['age']
        preg = request.form['preg']
        temp=preg
        if int(preg)>=2:
            preg = 2
        glu = request.form['glu']
        bp = request.form['bp']
        st = request.form['st']
        ins = request.form['ins']
        bmi = request.form['bmi']
        dbf = request.form['dbf']
        output = predict_diabetes(preg,glu,bp,st,ins,bmi,dbf,age)
        context = {'output':output,'age':age,'preg':preg,'glu':glu,'bp':bp,'st':st,'ins':ins,'bmi':bmi,'dbf':dbf,'diabetes':1}
        return render_template('results.html',context=context)

@app.route('/pneumonia',methods=['GET','POST'])
def pneumonia():
    if request.method=='GET':
        return render_template('pneumonia.html')
    else:

        user_file = request.files['user_img']
        path = './static/{}'.format(user_file.filename)
        user_file.save(path)
        '''output = predict_class(path)
        New'''


        img = image.load_img(request.files['user_img'], target_size=(256, 256))

        im2arr = np.array(img)
        im2arr = np.expand_dims(im2arr,axis=0)
        model = load_model('pneumonia_tf1.h5')
        result = model.predict(im2arr)
        output = list(result[0])


        if output==0.0:
            output='Normal'
        else:
            output='Pneumonia'
        context = {
            'prediction': output,
            'path':path


        }
        return render_template('results.html', context=context)





if __name__ == '__main__':
    app.run(debug=True)