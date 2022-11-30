from fnmatch import translate
from flask import Flask, redirect, url_for, render_template, request
import logic as l
import csv
import sys
import json
import pandas as pd
import os
import ast
IMG_FOLDER = os.path.join('static/', 'img/')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMG_FOLDER
@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'GET':
        institutions = []
        labels = []
        values = []
        tipos = ["ALL","CINDA","SMILE","OTHER"]
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
        u = ['../static/img/u1.jpg','../static/img/u2.jpg','../static/img/u5.jpg','../static/img/u6.jpg']
        lenguajess = ['Abkhaz Language', 'Afrikaans', 'Arabic', 'Araona Language', 'Armenian',
 'Ayacucho Quechua', 'Aymara language', 'Ayoreo language', 'Baure Language',
 'Belarusian', 'Bengali', 'Bokmål', 'Bosnian', 'Bulgarian', 'Burmese',
 'Canichana Language', 'Cavineña Language', 'Cayubaba Language',
 'Chakobo language', 'Chiquitano Language', 'Croatian', 'Czech', 'Danish',
 'Dari', 'Dutch', 'English', 'Estonian', 'Filipino language', 'Finnish',
 'French', 'Georgian', 'German', 'Greek', 'Guarani', 'Guaraní language',
 'Guarayu Language', 'Gujarati', 'Haitian Creole', 'Haitian French', 'Hebrew',
 'Hindi', 'Hungarian', 'Icelandic', 'Indonesian', 'Irish', 'Italian',
 'Itene Language', 'Itonama Language', 'Jamaican English', 'Japanese',
 'Kallawaya', 'Kazakh', 'Khoe languages', 'Khoisan languages', 'Korean',
 'Kyrgyz', 'Languages of South Africa', 'Leco Language', 'Lithuanian', 'Lule',
 'Luxembourgish', 'Machiguenga language', 'Malay', 'Malaysian language',
 'Maltese', 'Modern Hebrew', 'Modern Standard Arabic', 'Movima language',
 'Moxos language', 'Mòoré Language', 'Māori language', 'Nama',
 'Northern Sotho language', 'Norwegian', 'Nynorsk', 'Pacahuara language',
 'Pashto', 'Pauserna Language', 'Persian', 'Polish', 'Portuguese',
 'Puquina Language', 'Quechuan languages', 'Reyesano Language', 'Romanian',
 'Romansh', 'Russian', 'Saami', 'Salvadoran Spanish', 'Sanskrit', 'Serbian',
 'Singapore English', 'Singaporean Mandarin', 'Sinhala Language',
 'Sirionó language', 'Slovak', 'Slovene language', 'Sotho language',
 'South African English', 'South African Sign Language',
 'Southern Ndebele language', 'Spanish', 'Standard Algerian Berber',
 'Standard Chinese', 'Standard Moroccan Amazigh', 'Swahili', 'Swati Language',
 'Swedish', 'Tacana Language', 'Tamil', 'Tapieté Language', 'Telugu', 'Thai',
 'Toromono language', 'Tsimané Language', 'Tsonga', 'Tswana', 'Turkish',
 'Ukrainian', 'Urdu', 'Uru language', 'Uzbek', 'Venda Language', 'Vietnamese',
 'Wichí Lhamtés Nocten Language', 'Xhosa', 'Yaminahua Language',
 'Yine language', 'Yuqui Language', 'Yuracaré language', 'Zulu']
        lenU = [0,2]
        lenL = len(lenguajess)
        return render_template('index.html', labels=labels, values=values, lenI=len(institutions), institutions=institutions, lenC=len(labels), lenT=len(tipos), tipos=tipos, u=u, lenU = lenU, lenguajess = lenguajess, lenL = lenL)
    elif request.method == 'POST':
        full_filename1 = os.path.join(app.config['UPLOAD_FOLDER'], 'u1.jpg')
        full_filename2 = os.path.join(app.config['UPLOAD_FOLDER'], 'u5.jpg')
        full_filename3 = os.path.join(app.config['UPLOAD_FOLDER'], 'u6.jpg')
        full_filename4 = os.path.join(app.config['UPLOAD_FOLDER'], 'u4.jpg')
        a = [full_filename1,full_filename2,full_filename3,full_filename1]
        country = request.form['country']
        tipo = request.form['tipo']
        programa = request.form['program']
        lenguajes = request.form.getlist('lenguage[]')
        print('Esto son lenguajes------------------------------------------------------------------')
        print(lenguajes)
       
        promedio = int(request.form['promedio'])
        leng = json.dumps({"main": lenguajes})
        u = ['../static/img/u1.jpg','../static/img/u2.jpg','../static/img/u5.jpg','../static/img/u6.jpg']
        lenU = 4
        return redirect(url_for("tops",country = country, tipo = tipo, programa = programa, leng = leng, promedio = promedio, u= u, lenU = lenU))
       


     
@app.route('/tops', methods=['POST','GET'])
def tops():
    global tipo 
    global programa 
    global lenguajes 
    global country 
    global promedio
    global guarda
    global guardaCount
    if request.method == 'GET':
        tipo = request.args.get("tipo")
        guardaCount = 0
        programa = request.args.get("programa")
        
        leng = request.args.get("leng")
        
        leng2 = ast.literal_eval(leng)
        
        lenguajes = leng2["main"]
        print("va")
        country = request.args.get("country")
       
        promedio = int(request.args.get("promedio"))
        
        file = open('translateProgramas.json',encoding="utf8")
        translatePrograms = json.load(file)
        programa = translatePrograms[programa]
        
        #main(pCountry,pPrograma,pGpa,pPeso,pSeats)
        difi = l.sacarDificultad(programa)
        puestos = l.sacarPuestos(programa)
        r,contenedores, FiltrarSOI2, universidades, staywishes, SMILE, CINDA, F2 = l.man(country,programa,promedio,difi,puestos,lenguajes, "ALL")
        print("guardando")
        #guarda=l.guardarUni("Universität des Saarlandes", 0.86,["dg.guarin20@uniandes.edu.co", "Administration Bsc", "5", ["German","English"], []],FiltrarSOI2)
        guarda = l.guardarUni("",0.86,["dg.guarin20@uniandes.edu.co", "Administration Bsc", "5", ["German","English"], []],FiltrarSOI2, guardaCount)
        unis = l.alistarDatos(FiltrarSOI2, universidades)
        F3 = F2['Institution'].values
        print("a ver")
        print(unis.columns)
        print(FiltrarSOI2.columns)
        print("porfa")
        if guardaCount == 0:
            organizado = l.organizar(unis,guarda, guardaCount)
            guardaN = ["No hay"]
            guardaR = ["No hay"]
            guardaP = ["No hay"]
            lenG = len(guardaN)
        else: 
            recomendacionColl = l.SistemaRecomendacionCollab(staywishes, guarda, unis)
            guardaN, guardaR, guardaP = l.definirElGuardado(guarda)
            lenG = len(guardaN)
            guarda2 = l.alistarDatos2(FiltrarSOI2, recomendacionColl, "Ingles" )
            print("guarda")
            print(unis)
            recomendacionContent = l.SistemRecomendacionContent(unis, guarda2)
            print("recomendacionnnnnnnnnnnnnnnnnnnnnnnnnn")
            print(recomendacionContent)
            print(guarda)
            organizado = l.organizar(recomendacionContent, guarda)
            print("organizadooooooooooooooooooooooooooo")
            print(organizado)
        recomColl = organizado['Nombre'].values
        por = organizado["Porcentaje"].values
        porcentajeR = l.probabilidad(por)
        relationR = organizado["relation"].values
        relation = l.organizarRelation (recomColl, relationR)
        linkRecom = l.obtenerLink(unis, recomColl) 
        institutions_url = pd.read_excel('institution_url.xlsx')
        urls = []
        print(r)
        opciones = []
        full_filename1 = os.path.join(app.config['UPLOAD_FOLDER'], 'u1.jpg')
        full_filename2 = os.path.join(app.config['UPLOAD_FOLDER'], 'u5.jpg')
        full_filename3 = os.path.join(app.config['UPLOAD_FOLDER'], 'u6.jpg')
        full_filename4 = os.path.join(app.config['UPLOAD_FOLDER'], 'u4.jpg')
        a = [full_filename1,full_filename2,full_filename3,full_filename1]
        valoresR = [0,3,6]
        #a = ['u3.jpg','u2.jpg','u3.jpg','u3.jpg']
        if len(r) != 0:
            for i in r:
                tempK = list(i.keys())
                opciones.append(tempK[0])
                types = l.sacarTipo(unis, opciones)
                urls = l.obtenerLink(unis, opciones)
                #if not institutions_url[institutions_url['Name'] == tempK[0]].empty:
                 #   temp = institutions_url[institutions_url['Name'] == tempK[0]]
                  #  url = temp['URL'].values[0]
                   # urls.append(url)
                #else:
                 #   urls.append('NA')
        
            return render_template('top3.html',opciones=opciones,lenO=len(opciones),country=country,contenedores=contenedores,urls=urls,a=a, recomendacionColl = recomColl, types = types, por = porcentajeR, relation = relation, linkRecom = linkRecom, lenG= lenG, guardaN = guardaN, guardaR = guardaR, guardaP= guardaP, valoresR = valoresR)
    elif request.method == 'POST':
        guardaCount = 1
        file = open('translateProgramas.json',encoding="utf8")
        translatePrograms = json.load(file)
        
        
        #main(pCountry,pPrograma,pGpa,pPeso,pSeats)
        difi = l.sacarDificultad(programa)
        puestos = l.sacarPuestos(programa)
        r,contenedores, FiltrarSOI2, universidades, staywishes, SMILE, CINDA, F2 = l.man(country,programa,promedio,difi,puestos,lenguajes, "ALL")
        g = request.form['save_btn']
        print("revisar------")
        print(g)
        print("ojalaaa")
        
        guarda=l.guardarUni(g, 0.86, guarda, FiltrarSOI2, guardaCount)
        unis = l.alistarDatos(FiltrarSOI2, universidades)
        
        F3 = F2['Institution'].values
        recomendacionColl = l.SistemaRecomendacionCollab(staywishes, guarda, unis)
        guardaN, guardaR, guardaP = l.definirElGuardado(guarda)
        lenG = len(guardaN)
        guarda2 = l.alistarDatos2(FiltrarSOI2, recomendacionColl, "Ingles" )
        recomendacionContent = l.SistemRecomendacionContent(unis, guarda2)
        organizado = l.organizar(recomendacionContent, guarda, guardaCount)
        recomColl = organizado['Nombre'].values
        por = organizado["Porcentaje"].values
        porcentajeR = l.probabilidad(por)
        relationR = organizado["relation"].values
        relation = l.organizarRelation (recomColl, relationR)
        linkRecom = l.obtenerLink(unis, recomColl) 
        institutions_url = pd.read_excel('institution_url.xlsx')
        urls = []
        opciones = []
        full_filename1 = os.path.join(app.config['UPLOAD_FOLDER'], 'u1.jpg')
        full_filename2 = os.path.join(app.config['UPLOAD_FOLDER'], 'u5.jpg')
        full_filename3 = os.path.join(app.config['UPLOAD_FOLDER'], 'u6.jpg')
        full_filename4 = os.path.join(app.config['UPLOAD_FOLDER'], 'u4.jpg')
        valoresR = [0,3,6]
        a = [full_filename1,full_filename2,full_filename3,full_filename1]
        #a = ['u3.jpg','u2.jpg','u3.jpg','u3.jpg']
        if len(r) != 0:
            for i in r:
                tempK = list(i.keys())
                opciones.append(tempK[0])
                types = l.sacarTipo(unis, opciones)
                urls = l.obtenerLink(unis, opciones)
                #if not institutions_url[institutions_url['Name'] == tempK[0]].empty:
                 #   temp = institutions_url[institutions_url['Name'] == tempK[0]]
                  #  url = temp['URL'].values[0]
                   # urls.append(url)
                #else:
                 #   urls.append('NA')
        
            return render_template('top3.html',opciones=opciones,lenO=len(opciones),country=country,contenedores=contenedores,urls=urls,a=a, recomendacionColl = recomColl, types = types, por = porcentajeR, relation = relation, linkRecom = linkRecom, lenG= lenG, guardaN = guardaN, guardaR = guardaR, guardaP= guardaP, valoresR = valoresR)
    

if __name__ == '__main__':
    app.run(debug=True)
