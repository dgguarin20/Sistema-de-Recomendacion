# Importación de librerias
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report,make_scorer,recall_score
from sklearn.model_selection import ShuffleSplit,GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def cargarData():
    externalInstitutionsDF = pd.read_csv("externalInstitutions.csv",sep=',')
    stayOpportunityDF = pd.read_csv("relation.csv",sep=',')
    relationInstitution = pd.read_csv("relation_institution.csv",sep=',')
    programasUniandes = pd.read_csv("programaUniandes.csv",sep=',')
    staysAPI = pd.read_csv("stay.csv", sep=',')
    staysWishesAPI = pd.read_csv("movewishes.csv", sep=',')
    seatsDF = pd.read_csv("seats.csv", sep=',')
    unisClase0 = pd.read_excel("unisClase0.xlsx")
    unisClase1 = pd.read_excel("unisClase1.xlsx")
    cupos = pd.read_excel("cupos.xlsx")
    viajesExitosos = pd.read_excel('viajeExitosos.xlsx')
    
    return (externalInstitutionsDF,stayOpportunityDF,relationInstitution,programasUniandes,staysAPI,staysWishesAPI,seatsDF,unisClase0,unisClase1,cupos,viajesExitosos)
    

    
def dataPrepInic(externalInstitutionsDF,stayOpportunityDF,relationInstitution,programasUniandes):
    stayOpportunityDF = stayOpportunityDF.filter(['relation.level','relation.id', 'relation.name','relation.status.id','relation.relation_type.id','relation.course','relation.direction.id'])
    # DF queda con stay opportunities
    stayOpportunityDF = stayOpportunityDF[stayOpportunityDF['relation.relation_type.id'] == 4]

    # DF queda con outgoing
    stayOpportunityDF = stayOpportunityDF[stayOpportunityDF['relation.direction.id'] == 2]

    # DF queda con los que tienen status Active y silent
    stayOpportunityDF = stayOpportunityDF[stayOpportunityDF['relation.status.id'].isin([2,3])]

    # DF queda con los que son para pregrado
    stayOpportunityDF = stayOpportunityDF[~stayOpportunityDF['relation.level'].isin(['Postgraduate / Master','Doctorate / PhD','Doctorate / PhD, Short cycle'])]

    #stayOpportunityDF.info()
    relationInstitution = relationInstitution.filter(['relation_institution.relation.id','relation_institution.institution.id'])

    # Merge relation con relation Institution (Merge con tabla puente)
    SOI = pd.merge(stayOpportunityDF, relationInstitution, left_on='relation.id',right_on='relation_institution.relation.id', how='left')

    # Se hace ahora merge de SOI que es el resultado del merge anterior con las external Institutions (esta contiene toda la información necesaria de la institution)
    SOI = pd.merge(SOI, externalInstitutionsDF, left_on='relation_institution.institution.id',right_on='Institution: ID', how='left')

    # Al hacer merge en la columna Institution: ID  quedaron algunos nulos, esto significa que algunas stay opportunities tienen vinculada alguna institution que no estaba en la lista de institutions. Se procede a eliminar esos nulos
    SOI = SOI.dropna(axis=0, subset=['Institution: ID'])

    # Algunas columnas quedaron con los degree programas en nulo por lo tanto se eliminaran
    SOI = SOI.dropna(axis=0, subset=['relation.course'])

    # Se maneja el tema de nulos en las siguentes columnas
    SOI["Minimum GPA/4"].fillna("0", inplace = True)
    SOI["Language requirement 1"].fillna("No language requirement", inplace = True)
    SOI["Language requirement 2"].fillna("No language requirement", inplace = True)
    
    # Se procede a cambiarle el tipo a la columna Minimum GPA/4 ya que es tipo object y necesita ser float
    languageScores = ['A1','A2','B1','B2','C1','C2']
    gpasFloats = []
    for (columnName, columnData) in SOI.iteritems():
        if columnName == 'Minimum GPA/4':
            for gpa in columnData:
                if gpa not in languageScores:
                    gpa = gpa.replace(',','.')
                    gpasFloats.append(gpa)
                else:
                    gpasFloats.append('0')
            columnData = gpasFloats
                
            #print(columnData)
            SOI['Minimum GPA/4'] = columnData
            
    SOI["Minimum GPA/4"] = SOI["Minimum GPA/4"].astype(str).astype(float)

    # LIMPIAR PARA SACAR LOS QUE SON DE POS GRADO
    # NUMERO MAX DE PROGRAMAS EN PRE GRADO SON 45
    programasPregrado = programasUniandes[programasUniandes['Degree type'] == 'Bachelor']
    programasPregrado = programasPregrado['Name'].tolist()
    columnDPClean = []

    for index, row in SOI.iterrows():
        ans = []
        arrayDP = row['relation.course'].split('|| ')
        
        for dp in arrayDP:
            if dp in programasPregrado:        
                ans.append(dp+',')
                
        columnDPClean.append(''.join(ans))

    SOI['relation.course'] = columnDPClean

    del SOI['relation.level']
    del SOI['relation.name']
    del SOI['relation.status.id']
    del SOI['relation.relation_type.id']
    del SOI['relation.direction.id']
    del SOI['relation_institution.relation.id']
    del SOI['relation_institution.institution.id']

    # Sacar la cantidad total de programas que pueden participar
    dificultadI = []
    numeroMaxProgramas = 45
    lgnth = []
    count = 0
    sizes = []

    for index, row in SOI.iterrows():
        if pd.isnull(row['relation.course']):
            #print('No debio haber entrado')
            pass
        else:
            arrayDP = row['relation.course'].split(',')
            size = len(arrayDP)
            sizes.append(size)
            lgnth.append(size)
            difi = size/numeroMaxProgramas
            
        dificultadI.append(difi)

    SOI['Dificultad SO'] = dificultadI
    return SOI


def over4(gpaO5):
    return ((gpaO5*4)/5)

def validUniversitites(SOI,country,programa,gpa,peso,seats,lenguajes):
   
    #country, gpa, programa, peso, seats=  
    lenguajes.append('No language requirement')
    gpa = over4(gpa)
    filteredInstitutionsDF = SOI[(SOI['Country'] == country) & (SOI['Minimum GPA/4'] <= gpa) & (SOI['relation.course'].str.contains(programa))  & (SOI['Language requirement 1'].isin(lenguajes))]
    print('Esto son las universidades filtradas')
    print(filteredInstitutionsDF)
    return filteredInstitutionsDF, gpa, peso, seats



def ponerLabels(t):
    laInfo = {}
    stayIds = []
    for index, rowOut in t.iterrows():
        id = rowOut['Stay: ID']
        if id not in stayIds:
            stayIds.append(id)
        #laInfo[id]
        temp = {1:{'Status':None,'Promedio':None,'Puestos':None,'Dificultad':0,'Institution':None},2:{'Status':None,'Promedio':None,'Puestos':None,'Dificultad':0,'Institution':None},3:{'Status':None,'Promedio':None,'Puestos':None,'Dificultad':0,'Institution':None},4:{'Status':None,'Promedio':None,'Puestos':None,'Dificultad':0,'Institution':None}}
        tempDF = t[t['Stay: ID'] == id]
        count = 0
        for index,rowIn in tempDF.iterrows():
            temp[rowIn['Rank']]['Status'] = rowIn['Status selection']
            temp[rowIn['Rank']]['Promedio'] = rowIn['Stay: GPA outgoing']
            temp[rowIn['Rank']]['Puestos'] = rowIn['Number']
            temp[rowIn['Rank']]['Dificultad'] = rowIn['Dificultad SO']
            temp[rowIn['Rank']]['Institution'] = rowIn['Institution']
        
            laInfo[id] = temp
            count+=1

        #dfTrain = {'Promedio':None,'Puestos':None,'Label':None, 'Dificultad':0}
        promedios = []
        puestos = []
        dificultades = []
        instituciones = []
        labels = []
        for idS in stayIds:
            if laInfo[idS][1]['Status'] == 'Selected':
                promedios.append(laInfo[idS][1]['Promedio'])
                puestos.append(laInfo[idS][1]['Puestos'])
                dificultades.append(laInfo[idS][1]['Dificultad'])
                instituciones.append(laInfo[idS][1]['Institution'])
                labels.append(1)
            elif laInfo[idS][2]['Status'] == 'Selected':
                promedios.append(laInfo[idS][1]['Promedio'])
                puestos.append(laInfo[idS][1]['Puestos'])
                dificultades.append(laInfo[idS][1]['Dificultad'])
                instituciones.append(laInfo[idS][1]['Institution'])
                labels.append(0)

                promedios.append(laInfo[idS][2]['Promedio'])
                puestos.append(laInfo[idS][2]['Puestos'])
                dificultades.append(laInfo[idS][2]['Dificultad'])
                instituciones.append(laInfo[idS][2]['Institution'])
                labels.append(1)
            elif laInfo[idS][3]['Status'] == 'Selected':
                promedios.append(laInfo[idS][1]['Promedio'])
                puestos.append(laInfo[idS][1]['Puestos'])
                dificultades.append(laInfo[idS][1]['Dificultad'])
                instituciones.append(laInfo[idS][1]['Institution'])
                labels.append(0)

                promedios.append(laInfo[idS][2]['Promedio'])
                puestos.append(laInfo[idS][2]['Puestos'])
                dificultades.append(laInfo[idS][2]['Dificultad'])
                instituciones.append(laInfo[idS][2]['Institution'])
                labels.append(0)

                promedios.append(laInfo[idS][3]['Promedio'])
                puestos.append(laInfo[idS][3]['Puestos'])
                dificultades.append(laInfo[idS][3]['Dificultad'])
                instituciones.append(laInfo[idS][3]['Institution'])
                labels.append(1)
            else:
                
                promedios.append(laInfo[idS][1]['Promedio'])
                puestos.append(laInfo[idS][1]['Puestos'])
                dificultades.append(laInfo[idS][1]['Dificultad'])
                instituciones.append(laInfo[idS][1]['Institution'])
                labels.append(0)

                promedios.append(laInfo[idS][2]['Promedio'])
                puestos.append(laInfo[idS][2]['Puestos'])
                dificultades.append(laInfo[idS][2]['Dificultad'])
                instituciones.append(laInfo[idS][2]['Institution'])
                labels.append(0)

                promedios.append(laInfo[idS][3]['Promedio'])
                puestos.append(laInfo[idS][3]['Puestos'])
                dificultades.append(laInfo[idS][3]['Dificultad'])
                instituciones.append(laInfo[idS][3]['Institution'])
                labels.append(0)

    dataModel = {'Promedio':promedios,'Puestos':puestos,'Label':labels,'Dificultad':dificultades,'Institution':instituciones}
    dataForModel = pd.DataFrame(dataModel)
    dataForModel = dataForModel.dropna(axis=0,subset=['Promedio'])
    # Conversion a numerico
    languageScores = ['A1','A2','B1','B2','C1','C2']
    gpasFloats = []
    for (columnName, columnData) in dataForModel.iteritems():
        if columnName == 'Promedio':
            for gpa in columnData:
                if gpa not in languageScores:
                    gpa = gpa.replace(',','.')
                    gpasFloats.append(gpa)
                else:
                    gpasFloats.append('0')
            columnData = gpasFloats
                
            #print(columnData)
            dataForModel['Promedio'] = columnData



    dataForModel["Promedio"] = dataForModel["Promedio"].astype(str).astype(float)
    data4Model = dataForModel
    institutions = dataForModel['Institution'].unique()
    return (dataForModel,institutions)


def prepDataForModel(filteredInstitutionsDF,staysAPI,staysWishesAPI,seatsDF,cupos):

    cupos = cupos[cupos['Relation: Relation type'] == 'Stay opportunity']
    cupos = cupos[cupos['Relation: Status'] != 'Terminated']
    cupos = cupos[cupos['Relation: Status'] != 'Cancelled']
    cupos = cupos.dropna(axis=0,subset=['Relation: Level'])
    cupos = cupos[cupos['Relation: Level'].str.contains('Undergraduate')]

    del cupos['Relation: Relation type']
    del cupos['Relation: Status']
    del cupos['Relation: Level']
    del cupos['Relation: Direction']

    tasas = {}
    for index,rowIn in cupos.iterrows():
        if rowIn['Relation: Relation ID'] not in tasas.keys():
            if rowIn['Number'] == rowIn['Remaining seats']:
              
                tasas[rowIn['Relation: Relation ID']] =  0.0001
            elif rowIn['Remaining seats'] == 0:
           
                tasas[rowIn['Relation: Relation ID']] =  1
            else:
                
                tasas[rowIn['Relation: Relation ID']] =  1 - (rowIn['Remaining seats'] / rowIn['Number'])
        else:
            if rowIn['Number'] == rowIn['Remaining seats']:
                p =  0.0001
            elif rowIn['Remaining seats'] == 0:
                p =  1
            else:
                p = 1 - (rowIn['Remaining seats'] / rowIn['Number'])
            
            if  tasas[rowIn['Relation: Relation ID']] * p < 0:
                #print('Menor a cero')
                #print('Valor anterior')
                #print(tasas[rowIn['Relation: Relation ID']])
                #print('Valor despues')
                #print(tasas[rowIn['Relation: Relation ID']] * p)
                tasas[rowIn['Relation: Relation ID']] = tasas[rowIn['Relation: Relation ID']] * p
            elif tasas[rowIn['Relation: Relation ID']] * p > 1:
                #print('Menor a uno')
                #print('Valor anterior')
                #print(tasas[rowIn['Relation: Relation ID']])
                #print('Valor despues')
                #print(tasas[rowIn['Relation: Relation ID']] * p)
                tasas[rowIn['Relation: Relation ID']] = tasas[rowIn['Relation: Relation ID']] * p

    #print('Estos son tasas')
    #print(tasas.keys())
    #print(len(tasas))

    
    instiPeso = filteredInstitutionsDF.filter(['relation.id','Dificultad SO'])
    
    instiPeso = instiPeso.drop_duplicates()
    #print('Insti peso antes')
    #print(instiPeso)

    columnaNueva = []
    for index,rowIn in instiPeso.iterrows():
        if rowIn['relation.id'] in tasas.keys():
            columnaNueva.append(rowIn['Dificultad SO'] * tasas[rowIn['relation.id']])
        else:
            columnaNueva.append(rowIn['Dificultad SO'])

    instiPeso['Dificultad SO'] = columnaNueva
    #print('Insti peso despues')
    #print(instiPeso)
    
    #assert False,'breakpoint'
    #print('-- Insti Peso --')
    #print(instiPeso)
    #stayOportunityIds = instiPeso['relation.id'].unique()
    #print('-- len Insti peso --')
    #print(instiPeso)

    #print('-- len Insti peso unique --')
    #print(len(instiPeso['Name'].unique()))
    #print('len stay wishes caso')
    #print(len(staysWishesAPI[staysWishesAPI['Stay: ID']== 2833]))
    staysWishesAPI = staysWishesAPI[staysWishesAPI['Status selection']!= 'Pending']
    SWI = pd.merge(staysWishesAPI, instiPeso, left_on='Relation: ID',right_on='relation.id')
    #print('len swi caso')
    #print(len(SWI[SWI['Stay: ID'] == 2833]))
    SWI['Start period'] = SWI['Start period'].astype(str)
    SWI['Start period'] = SWI['Start period'].str.strip()

    seatsDF['Academic period'] = seatsDF['Academic period'].astype(str)
    seatsDF['Academic period'] = seatsDF['Academic period'].str.strip()

    t = pd.merge(SWI, seatsDF,  left_on=['Start period','Relation: ID'], right_on = ['Academic period','Relation: Relation ID'])
    del t['Academic period']
    del t['Start period']
    del t['Relation: ID']
    del t['Stay opportunity']
    

    t = t.dropna(axis=0,subset=['Stay: ID'])
    t = t.dropna(axis=0,subset=['Stay: GPA outgoing'])
    #print('--- Esto es t ---')
    #print(t)
    ans = ponerLabels(t)
    return ans

def averageModelo(recalls):
    sumR = 0
    for r in recalls:
        sumR += r
    averageRecalls = sumR/len(recalls)
    return averageRecalls
    #print(f'{averageRecalls*100}%')

def elModeloKNN(dataForModel,ins,prom,peso,seats,unisClase0,unisClase1):
    unisName0 = unisClase0['Name'].unique()
    unisName1 = unisClase1['Name'].unique()
    """
    print(len(unisName1))
    print(len(unisName0))
    print('----- Unis 1 -----')
    print(unisName1)
    print('----- Unis 0 -----')
    print(unisName0)
    print('----- Institutions que entran al modelo -----')
    #print(ins)
    """
    train=dataForModel.sample(frac=0.8,random_state=200)
    #test=dataForModel.drop(train.index)
    scoreres = {'recall_score': make_scorer(recall_score)}
    puntajes = {}
    unisCero = []
    unisUno = []
    recalls = []
    for u in ins:
        tTrain1 = train
        #tTrain2 = train
        entra = False
        entra2 = False
        # Filtro la universidad u
        #print('Longitud de tu sabes')
        #r = len(tTrain1['Institution'] == u)
        #print(r)
        details = tTrain1.apply(lambda x : True if x['Institution'] == u else False, axis = 1)
        count = len(details[details == True].index)
        if count < 10:
            print('--- Entro a caso poco ---')
            # Tengo que poblar
            if u in unisName0:
                print('-- 0 tTrain1 (Antes) --')
                print(len(tTrain1))
                tTrain1 = tTrain1[tTrain1['Institution'].isin(unisName0)]
                print('-- tTrain despues --')
                print(len(tTrain1))
            elif u in unisName1:
                #print(tTrain2)
                print('-- 1 tTrain1 (Antes) --')
                print(len(tTrain1))
                tTrain1 = tTrain1[tTrain1['Institution'].isin(unisName1)]
                print('-- tTrain despues --')
                print(len(tTrain1))
            else:
                # NO lo podemos corregir
                entra2 =True
                print('No tiene remedio')
        else:
             tTrain1 = tTrain1[tTrain1['Institution'] == u]
        if not entra2:
            minimum = (min(len(tTrain1[tTrain1['Label'] == 1]),len(tTrain1[tTrain1['Label'] == 0])), '0' if len(tTrain1[tTrain1['Label'] == 1]) > len(tTrain1[tTrain1['Label'] == 0]) else '1')
            print('-------- UNIVERSIDAD ---------')
            print(f'Esto es u:{u}')
            details0 = tTrain1.apply(lambda x : True if x['Label'] == 0 else False, axis = 1)
            details1 = tTrain1.apply(lambda x : True if x['Label'] == 1 else False, axis = 1)
            cerosLabel = len(details0[details0 == True].index)
            onesLabel = len(details1[details1 == True].index)
            print(f'Esto es distribución labels: [1: {onesLabel},0: {cerosLabel}]')
            #tTest = test
            #tTest = tTest[tTest['Institution'] == u]

            # Particiono datos
            del tTrain1['Institution']
            yTrain = tTrain1['Label']
            del tTrain1['Label']
            xTrain = tTrain1

        

            scalar = preprocessing.StandardScaler().fit(xTrain)
            xTrain = scalar.transform(xTrain)
            #xTest = scalar.transform(xTest)

            # Creo modelo inicial
            modeloEuc = KNeighborsClassifier()
            modeloEuc.fit(xTrain, yTrain)

            # Hayo el k que me maximize el recall
            skf = ShuffleSplit(n_splits=10)
            if minimum[0] == 1:
                k_range = list(range(1, 2))
                entra = True

            elif minimum[0] == 0:
                pass
                #if minimum[1] == '0':
                    #unisCero.append({u:0})
                #else:
                    #unisUno.append({u:0})
            else:
                k_range = list(range(1, minimum[0]))
                entra = True
                
            if entra:
                print('Esta entrando a buscar el K')
                param_grid = dict(n_neighbors=k_range)
                random_search = GridSearchCV(modeloEuc, cv=skf, scoring=scoreres,param_grid=param_grid, verbose= 0,refit='recall_score')
                random_search.fit(xTrain,yTrain)

                #print('Recall score - ' + str(random_search.score(xTrain,yTrain)))
                recalls.append(random_search.score(xTrain,yTrain))
                print('-- SCORE --')
                print(random_search.score(xTrain,yTrain))
                #print('Test score - ' + str(random_search.score(xTest,yTest)))

                # Obtengo k
                k = random_search.best_params_["n_neighbors"]
                print(f'Esto es k: {k}')
                #print(f'Esto es k: {k}')
                modeloEuc = KNeighborsClassifier(n_neighbors=k)
                modeloEuc.fit(xTrain, yTrain)

                lab = modeloEuc.predict([[prom,seats,peso]])[0]
                print(f'Este es el label que le predijo {lab}')
                if lab == 0:
                    print(modeloEuc.predict_proba([[prom,seats,peso]]))
                    prob = modeloEuc.predict_proba([[prom,seats,peso]])[0][0]
                    print(f'Esta es la prob {prob}')
                    unisCero.append({u:prob})
                else:
                    print(modeloEuc.predict_proba([[prom,seats,peso]]))
                    prob = modeloEuc.predict_proba([[prom,seats,peso]])[0][1]
                    print(f'Esta es la prob {prob}')
                    unisUno.append({u:prob})

    puntajes[0] = unisCero
    puntajes[1] = unisUno 

    #avgRecall = averageModelo(recalls)

    return (puntajes[1])


def KNN(dataForModel,ins,viajesE,prom,peso,seats,unisClase0,unisClase1):
    #unisName0 = unisClase0['Name'].unique()
    #unisName1 = unisClase1['Name'].unique()
    #print(ins)
    """
    print(len(unisName1))
    print(len(unisName0))
    print('----- Unis 1 -----')
    print(unisName1)
    print('----- Unis 0 -----')
    print(unisName0)
    print('----- Institutions que entran al modelo -----')
    #print(ins)
    """
    #train=dataForModel.sample(frac=0.8,random_state=200)
    #test=dataForModel.drop(train.index)
    #scoreres = {'recall_score': make_scorer(recall_score)}
    puntajes = {}
    unisCero = []
    unisUno = []
    accuracys = []
    for u in ins:
        datosTemp = dataForModel
        #tTrain2 = train
        entra = False
        entra2 = False
        # Filtro la universidad u
        #print('Longitud de tu sabes')
        #r = len(tTrain1['Institution'] == u)
        #print(r)
        details = datosTemp.apply(lambda x : True if x['Institution'] == u else False, axis = 1)
        count = len(details[details == True].index)
        if count <= 30:
            #print('Menor a 30')
            tempViajes = viajesE
            entra2 = True
            #if u in (tempViajes['Institutions'].tolist()):
            registro = tempViajes[tempViajes['Institutions'] == u]
            if not registro.empty:
                #print('Tiene tasa')
                #print('--- Esto es registro ---')
                #print(registro)
                tasa = registro['Tasa Viaje Exitoso'].tolist()
                #print('--- Esto es tasa ---')
                #print(tasa)
                if tasa[0]+0.25 >= 0.2:
                    unisUno.append({u:tasa[0]+0.25})
                else:
                    unisCero.append({u:tasa[0]})
            else:
                print('NO tiene tasa')
        else:
             datosTemp = datosTemp[datosTemp['Institution'] == u]

        if not entra2:
            minimum = (min(len(datosTemp[datosTemp['Label'] == 1]),len(datosTemp[datosTemp['Label'] == 0])), '0' if len(datosTemp[datosTemp['Label'] == 1]) > len(datosTemp[datosTemp['Label'] == 0]) else '1')
            #print('-------- UNIVERSIDAD ---------')
            #print(f'Esto es u:{u}')
            details0 = datosTemp.apply(lambda x : True if x['Label'] == 0 else False, axis = 1)
            details1 = datosTemp.apply(lambda x : True if x['Label'] == 1 else False, axis = 1)
            cerosLabel = len(details0[details0 == True].index)
            onesLabel = len(details1[details1 == True].index)
            #print(f'Esto es distribución labels antes SMOTE: [1: {onesLabel},0: {cerosLabel}]')
            #tTest = test
            #tTest = tTest[tTest['Institution'] == u]
             
            # Particiono datos
            del datosTemp['Institution']
            yTrain = datosTemp['Label']
            del datosTemp['Label']
            xTrain = datosTemp

            # SMOTE
            oversample = SMOTE()
            xTrain,yTrain = oversample.fit_resample(xTrain,yTrain)
            counter = Counter(yTrain)
            print('Esto es el nuevo counter')
            print(counter)


            scalar = preprocessing.StandardScaler().fit(xTrain)
            xTrain = scalar.transform(xTrain)
            #xTest = scalar.transform(xTest)

            # Creo modelo inicial
            modeloEuc = KNeighborsClassifier()
            modeloEuc.fit(xTrain, yTrain)

            # Hayo el k que me maximize el recall
            skf = ShuffleSplit(n_splits=10)
            if minimum[0] == 1:
                k_range = list(range(1, 2))
                entra = True

            elif minimum[0] == 0:
                pass
                #if minimum[1] == '0':
                    #unisCero.append({u:0})
                #else:
                    #unisUno.append({u:0})
            else:
                k_range = list(range(1, minimum[0]))
                entra = True
                
            if entra:
                print('Esta entrando a buscar el K')
                param_grid = dict(n_neighbors=k_range)
                #random_search = GridSearchCV(modeloEuc, cv=skf, scoring=scoreres,param_grid=param_grid, verbose= 0,refit='recall_score')
                grid_search = GridSearchCV(modeloEuc, cv=skf,param_grid=param_grid, verbose= 0)
                grid_search.fit(xTrain,yTrain)

                #print('Recall score - ' + str(random_search.score(xTrain,yTrain)))
                #recalls.append(random_search.score(xTrain,yTrain))
                #print('-- SCORE --')
                #print(random_search.score(xTrain,yTrain))
                #print('Test score - ' + str(random_search.score(xTest,yTest)))

                # Obtengo k
                k = grid_search.best_params_["n_neighbors"]
                accuracy = grid_search.best_score_
                accuracys.append(accuracy)
                print(f'Esto es k: {k}')
                print(f'Esto es accuracy sin SMOTE: {accuracy}')
                modeloEuc = KNeighborsClassifier(n_neighbors=k)
                modeloEuc.fit(xTrain, yTrain)

                lab = modeloEuc.predict([[prom,seats,peso]])[0]
                print(f'Este es el label que le predijo {lab}')
                if lab == 0:
                    print(modeloEuc.predict_proba([[prom,seats,peso]]))
                    prob = modeloEuc.predict_proba([[prom,seats,peso]])[0][0]
                    print(f'Esta es la prob {prob}')
                    unisCero.append({u:prob})
                else:
                    print(modeloEuc.predict_proba([[prom,seats,peso]]))
                    prob = modeloEuc.predict_proba([[prom,seats,peso]])[0][1]
                    print(f'Esta es la prob {prob}')
                    unisUno.append({u:prob})

    puntajes[0] = unisCero
    puntajes[1] = unisUno 

    #avgRecall = averageModelo(accuracys)

    return (puntajes[1])




    

    

def top4(predicciones):
    #print('----- Predicciones -----')
    #print(predicciones)
    a = []
    top4 = []
    ans = []
    for p in predicciones:
        n = list(p.values())
        #print(n[0])
        a.append(n[0])
        

    #print(a)
    #print(type(a))
    a = np.array(a)
    #print(a)
    #r = np.where(a == 1,0,a)
    #print(f'len(r): {len(a)}')
    if len(a) >= 4:
        print('Args sort mayor a 4')
        print(a)
        ind = a.argsort()[-4:][::-1]
        print(ans)
    elif len(a) < 4:
        #for i in range(len(a)):
        #    ans.append(i)
        # Tengo indices ordenados
        print('Args sort menor a 4')
        print(a)
        ind = a.argsort()[::-1]
        print(ans)
    predicciones = np.array(predicciones)
    top4 = predicciones[ind]
    print('--- TOP 4 ---')
    print(top4)

    #print('--- Ans --- ')
    #print(ans)

    #for ind in range(len(predicciones)):
    #    if ind in ans:
    #        top4.append(predicciones[ind])

    return top4

def main(pCountry,pPrograma,pGpa,pPeso,pSeats):
    externalInstitutionsDF,stayOpportunityDF,relationInstitution,programasUniandes,staysAPI,staysWishesAPI,seatsDF,unisClase0,unisClase1,cupos = cargarData()
    SOI = dataPrepInic(externalInstitutionsDF,stayOpportunityDF,relationInstitution,programasUniandes)
    filteredInstitutionsDF,gpa,peso,seats = validUniversitites(SOI,pCountry,pPrograma,pGpa,pPeso,pSeats)
    #print(filteredInstitutionsDF)
    t = prepDataForModel(filteredInstitutionsDF,staysAPI,staysWishesAPI,seatsDF,cupos)
    dataForModel = t[0]
    institutions = t[1]
    #print('--- Data For Modelo ---')
    #print(dataForModel)
    #print('--- Institutions ---')
    #print(institutions)
    universidadesPrediccion, avgRecall = elModeloKNN(dataForModel,institutions,gpa,peso,seats,unisClase0,unisClase1)
    #print(universidadesPrediccion,avgRecall)
    r = top4(universidadesPrediccion)
    #print(f'Las 3 universidades a la que es más probable entrar son {r}')
    #print(f'Con un recall: {avgRecall}')
    #print(top3)
    return r

def sacarDificultad(programa):
    dificultades = pd.read_excel('dificultadPorPograma.xlsx')
    
    registro = dificultades[dificultades['Programa'] == programa]  
    temp = list(registro['Dificultad Promedio'])
    print('---------- Esto es temp ----------')
    print(temp)
    return temp[0]

def sacarPuestos(programa):
    puestos = pd.read_excel('seatsPorPogrmaa.xlsx')
    registro = puestos[puestos['Programa'] == programa]  
    temp = list(registro['Puestos Promedio'])
    return temp[0]

def man(pCountry,pPrograma,pGpa,pPeso,pSeats,pLenguajes):
    externalInstitutionsDF,stayOpportunityDF,relationInstitution,programasUniandes,staysAPI,staysWishesAPI,seatsDF,unisClase0,unisClase1,cupos,viajesE = cargarData()
    SOI = dataPrepInic(externalInstitutionsDF,stayOpportunityDF,relationInstitution,programasUniandes)
    filteredInstitutionsDF,gpa,peso,seats = validUniversitites(SOI,pCountry,pPrograma,pGpa,pPeso,pSeats,pLenguajes)
    #print(filteredInstitutionsDF)
    t = prepDataForModel(filteredInstitutionsDF,staysAPI,staysWishesAPI,seatsDF,cupos)
    dataForModel = t[0]
    institutions = t[1]
    #print('--- Data For Modelo ---')
    #print(dataForModel)
    #print('--- Institutions ---')
    #print(institutions)
    
    universidadesPrediccion = KNN(dataForModel,institutions,viajesE,gpa,peso,seats,unisClase0,unisClase1)
    #print(universidadesPrediccion,avgRecall)
    r = top4(universidadesPrediccion)
    contenedores = []
    for i in range(0, len(r)):
        prob = list(r[i].values())
        prob = prob[0]
        if prob >= 0 and prob < 0.3:
            contenedores.append('Baja')
        elif prob >= 0.3 and prob < 0.7:
            contenedores.append('Media')
        else:
            contenedores.append('Alta')

    print(f'Las 3 universidades a la que es más probable entrar son {r}')
    #print(f'Con un recall: {avg}')
    return r,contenedores





    