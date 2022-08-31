from fnmatch import translate
from flask import Flask, render_template, request
import logic as l
import csv
import sys
import json
import pandas as pd
import os
IMG_FOLDER = os.path.join('static/', 'img/')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMG_FOLDER
@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'GET':
        institutions = []
        labels = []
        values = []
        
        with open('costOfLiving_2022.csv',encoding="utf8") as file:
            csv_reader = csv.reader(file, delimiter=',')

            for row in csv_reader:
                labels.append(row[0])
                values.append(row[1])

        with open('programasUniandes.csv',encoding="utf8") as file:
            csv_reader = csv.reader(file, delimiter=',')

            for row in csv_reader:
                institutions.append(row[2])
        
        #print('----- Esta son labels -----')
        #print(labels, file=sys.stderr)
        #print('----- Esta son values ----')
        #print(values, file=sys.stderr)

        return render_template('index.html', labels=labels, values=values, lenI=len(institutions), institutions=institutions, lenC=len(labels))
    elif request.method == 'POST':
        full_filename1 = os.path.join(app.config['UPLOAD_FOLDER'], 'u1.jpg')
        full_filename2 = os.path.join(app.config['UPLOAD_FOLDER'], 'u5.jpg')
        full_filename3 = os.path.join(app.config['UPLOAD_FOLDER'], 'u6.jpg')
        full_filename4 = os.path.join(app.config['UPLOAD_FOLDER'], 'u4.jpg')
        a = [full_filename1,full_filename2,full_filename3,full_filename1]
        country = request.form['country']
        programa = request.form['program']
        lenguajes = []
        if 'esItaliano' in request.form:
            lenguajes.append('Italian')
        if 'esIngles' in request.form:
            lenguajes.append('English')
        if 'esFrances' in request.form:
            lenguajes.append('French')
        if 'esPortugues' in request.form:
            lenguajes.append('Portuguese')
        if 'esAleman' in request.form:
            lenguajes.append('German')

        print('Esto son lenguajes')
        print(lenguajes)
        
       
        #test = request.form['first_checkbox']
        file = open('translateProgramas.json',encoding="utf8")
        translatePrograms = json.load(file)
        programa = translatePrograms[programa]
        promedio = int(request.form['promedio'])
        #main(pCountry,pPrograma,pGpa,pPeso,pSeats)
        difi = l.sacarDificultad(programa)
        puestos = l.sacarPuestos(programa)
        r,contenedores = l.man(country,programa,promedio,difi,puestos,lenguajes)
        institutions_url = pd.read_excel('institution_url.xlsx')
        urls = []
        print(r)
        opciones = []
        #a = ['u3.jpg','u2.jpg','u3.jpg','u3.jpg']
        if len(r) != 0:
            for i in r:
                tempK = list(i.keys())
                opciones.append(tempK[0])
                if not institutions_url[institutions_url['Name'] == tempK[0]].empty:
                    temp = institutions_url[institutions_url['Name'] == tempK[0]]
                    url = temp['URL'].values[0]
                    urls.append(url)
                else:
                    urls.append('NA')
        
               

            return render_template('top3.html',opciones=opciones,lenO=len(opciones),country=country,contenedores=contenedores,urls=urls,a=a)
        else:
            print('Nada de nada')

       




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
    