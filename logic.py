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
from copy import deepcopy
from numpy import ndarray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def unisClaseViajes(externalI, staywishes):
    print(externalI['Minimum GPA/4'].unique())
    unisClase0 = externalI[externalI['Minimum GPA/4'].isna()]
    unisClase1 = externalI[~externalI['Minimum GPA/4'].isna()]
    wishes = staywishes[['Stay: Institution','Stay: Status', 'Stay: ID']]
    print(wishes['Stay: Status'].unique())
    percentaje = []
    nombre = []
    for i in range(0, len(unisClase0)):
        w = wishes[wishes['Stay: ID']==unisClase0['Institution: ID'].iloc[i]]
        completed = w[w['Stay: Status']=="Completed"]
        porcentaje = 0
        nombre.append(unisClase0['Name'].iloc[i])
        if len(w) == 0:
            percentaje.append(porcentaje)
        else:
            porcentaje = len(completed)/len(w)
            percentaje.append(porcentaje)
    for i in range(0, len(unisClase1)):
        w = wishes[wishes['Stay: ID']==unisClase1['Institution: ID'].iloc[i]]
        completed = w[w['Stay: Status']=="Completed"]
        porcentaje = 0
        nombre.append(unisClase1['Name'].iloc[i])
        if len(w) == 0:
            percentaje.append(porcentaje)
        else:
            porcentaje = len(completed)/len(w)
            percentaje.append(porcentaje)
           
    TasaViaje = pd.DataFrame()
    TasaViaje['Tasa Viaje Exitoso']= percentaje
    TasaViaje['Institutions'] = nombre
    return unisClase0, unisClase1, TasaViaje

def cargarData():
    externalInstitutionsDF = pd.read_csv("externalInstitutions.csv",sep=',')
    print(externalInstitutionsDF.columns)
    externalInstitutions = pd.read_csv("./Info/Institutions.csv",sep=',')
    externalI = externalInstitutions[['Institution: ID', 'Country', 'City', 'Name', 'Language requirement 1',
       'Language cerf score 1', 'Language requirement 2',
       'Language cerf score 2', 'Minimum GPA/4','Official Language']]
    externalI2 = externalI[externalI['Name'].notnull()]
    
    #-------
    stayOpportunityDF = pd.read_csv("relation.csv",sep=',')
    stayOpportunity = pd.read_csv("./Info/relationstayOpportunities.csv",sep=',')
    stOpp = stayOpportunity[['Level','Relation ID','Name','Status','Degree programme', 'Direction', 'Relation type','Frameworks']]
   
    #--------
    programasUniandes = pd.read_csv("programaUniandes.csv",sep=',')
    #--------
    staysAPI = pd.read_csv("stay.csv", sep=',')
    #--------
    staysWishesAPI = pd.read_csv("movewishes.csv", sep=',')
    staywishes = pd.read_csv("./Info/staywishesoutgoing.csv", sep= ',')
    
    stWishes = staywishes[['Stay: GPA outgoing','Start period', 'Rank','Frameworks', 'Relation: ID', 'Stay opportunity', 'Status selection', 'Institution','Stay wish ID']]
   
    #---------
    
    relationInstitution = pd.read_csv("relation_institution.csv",sep=',')
    relationI = staywishes[['Relation: ID', 'Institution' ]]
    relationI2 = relationI[relationI['Institution'].notnull()]
    #se puede eliminar y usar Name
     #--------
    seatsDF = pd.read_csv("seats.csv", sep=',')
    seats = pd.read_csv("./Info/flow.csv", sep=',')
    seat = seats[['Relation: Relation ID','Academic period','Number']]
    #+++++++++
    unisClase0 = pd.read_excel("unisClase0.xlsx")
    
    unisClase1 = pd.read_excel("unisClase1.xlsx")
    
    uniClase0, uniClase1, TasaViaje = unisClaseViajes(externalI,staywishes)
    #-----------
    cupos = pd.read_excel("cupos.xlsx")
    cupo = seats[['Number','Relation: Relation type', 'Relation: Direction', 'Relation: Status', 'Relation: Level', 'Relation: Relation ID', 'Remaining seats']]
    
    staywishes = pd.read_csv("staywishesoutgoinglast.csv",sep=',')
    staywishes['Form'].fillna("")
    
    return (externalI2,stOpp,relationI2,programasUniandes,staysAPI,stWishes,seat,uniClase0,uniClase1,cupo,TasaViaje, staywishes)

    
def dataPrepInic(externalInstitutionsDF,stayOpportunityDF,relationInstitution,programasUniandes):
    # DF queda con stay opportunities
    
    stayOpportunityDF = stayOpportunityDF[stayOpportunityDF['Relation type'] == 'Stay opportunity']

    # DF queda con outgoing
    stayOpportunityDF = stayOpportunityDF[stayOpportunityDF['Direction'] == 'Outgoing']

    # DF queda con los que tienen status Active y silent
    stayOpportunityDF = stayOpportunityDF[(stayOpportunityDF['Status'] == 'Silent') | (stayOpportunityDF['Status'] == 'Active')]

    # DF queda con los que son para pregrado
    stayOpportunityDF = stayOpportunityDF[~stayOpportunityDF['Level'].isin(['Postgraduate / Master','Doctorate / PhD','Doctorate / PhD, Short cycle'])]

    #stayOpportunityDF.info()
   # relationInstitution = relationInstitution.filter(['Relation ID','relation_institution.institution.id'])

    # Merge relation con relation Institution (Merge con tabla puente)
    SOI = pd.merge(stayOpportunityDF, relationInstitution, left_on='Relation ID',right_on='Relation: ID', how='left')

    # Se hace ahora merge de SOI que es el resultado del merge anterior con las external Institutions (esta contiene toda la información necesaria de la institution)
    SOI = pd.merge(SOI, externalInstitutionsDF, left_on='Institution',right_on='Name', how='left')
    print(externalInstitutionsDF['Name'].head(5))
    # Al hacer merge en la columna Institution: ID  quedaron algunos nulos, esto significa que algunas stay opportunities tienen vinculada alguna institution que no estaba en la lista de institutions. Se procede a eliminar esos nulos
    #SOI = SOI.dropna(axis=0, subset=['Institution: ID'])

    # Algunas columnas quedaron con los degree programas en nulo por lo tanto se eliminaran
    #SOI = SOI.dropna(axis=0, subset=['relation.course'])

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
    print("----")
    print(SOI['Degree programme'].isnull().sum())
    print(len(SOI))
    print("----")
    SOI3 = SOI[SOI['Degree programme'].notnull()]
    for index, row in SOI3.iterrows():
        ans = []
        arrayDP = row['Degree programme'].split('|| ')
        
        for dp in arrayDP:
            if dp in programasPregrado:        
                ans.append(dp+',')
                
        columnDPClean.append(''.join(ans))
   
    SOI3['Degree programme'] = columnDPClean
    SOI3 = SOI3.dropna(axis=0,subset=['Institution'])
    SOI2 = deepcopy(SOI3)
    print(SOI3.columns)
   
    del SOI3['Level']
    del SOI3['Name_x']
    del SOI3['Status']
    del SOI3['Relation type']
    del SOI3['Direction']
    del SOI3['Relation ID']
    #del SOI['relation_institution.institution.id']
    
    # Sacar la cantidad total de programas que pueden participar
    dificultadI = []
    numeroMaxProgramas = 45
    lgnth = []
    count = 0
    sizes = []

    for index, row in SOI3.iterrows():
        if pd.isnull(row['Degree programme']):
            #print('No debio haber entrado')
            pass
        else:
            arrayDP = row['Degree programme'].split(',')
            size = len(arrayDP)
            sizes.append(size)
            lgnth.append(size)
            difi = size/numeroMaxProgramas
            
        dificultadI.append(difi)

    SOI3['Dificultad SO'] = dificultadI
    SOI2['Dificultad SO'] = dificultadI
    return SOI3, SOI2


def over4(gpaO5):
    return ((gpaO5*4)/5)

def validUniversitites(SOI,country,programa,gpa,peso,seats,lenguajes, SOI2):
   
    #country, gpa, programa, peso, seats=  
    lenguajes.append('No language requirement')
    print(lenguajes)
    gpa = over4(gpa)
    if(country == "all"):
        filteredInstitutionsDF = SOI[(SOI['Minimum GPA/4'] <= gpa) & (SOI['Degree programme'].str.contains(programa))  & (SOI['Language requirement 1'].isin(lenguajes))]
    else:                                      
        filteredInstitutionsDF = SOI[(SOI['Country'] == country) & (SOI['Minimum GPA/4'] <= gpa) & (SOI['Degree programme'].str.contains(programa))  & (SOI['Language requirement 1'].isin(lenguajes))]
    
    if(country == "all"):
        filteredInstitutionsDF2 = SOI2[(SOI2['Minimum GPA/4'] <= gpa) & (SOI2['Degree programme'].str.contains(programa))  & (SOI2['Language requirement 1'].isin(lenguajes))]
        print("fdskalfmsadklmfklasdjfklasdjfklasdjkflasjdklfjadskljfklasdjflkasdjflkasdjfklasd")
        print(filteredInstitutionsDF2["Country"].unique())
        print(filteredInstitutionsDF2["Institution"].unique())
        print(filteredInstitutionsDF2["Language requirement 1"].unique())
    else:                                      
        filteredInstitutionsDF2 = SOI2[(SOI['Country'] == country) & (SOI2['Minimum GPA/4'] <= gpa) & (SOI2['Degree programme'].str.contains(programa))  & (SOI2['Language requirement 1'].isin(lenguajes))]
    
    #print('Esto son las universidades filtradas')
    
    
    return filteredInstitutionsDF, gpa, peso, seats, filteredInstitutionsDF2


def ponerLabels(t):
    laInfo = {}
    stayIds = []
    promedios = []
    puestos = []
    dificultades = []
    instituciones = []
    labels = []
    print(t.columns)
    for index, rowOut in t.iterrows():
        id = rowOut['Stay wish ID']
        if id not in stayIds:
            stayIds.append(id)
        #laInfo[id]
        temp = {1:{'Status':None,'Promedio':None,'Puestos':None,'Dificultad':0,'Institution':None},2:{'Status':None,'Promedio':None,'Puestos':None,'Dificultad':0,'Institution':None},3:{'Status':None,'Promedio':None,'Puestos':None,'Dificultad':0,'Institution':None},4:{'Status':None,'Promedio':None,'Puestos':None,'Dificultad':0,'Institution':None}}
        tempDF = t[t['Stay wish ID'] == id]
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

    
    instiPeso = filteredInstitutionsDF.filter(['Relation: ID','Dificultad SO'])
    
    instiPeso = instiPeso.drop_duplicates()
    #print('Insti peso antes')
    #print(instiPeso)

    columnaNueva = []
    for index,rowIn in instiPeso.iterrows():
        if rowIn['Relation: ID'] in tasas.keys():
            columnaNueva.append(rowIn['Dificultad SO'] * tasas[rowIn['Relation: ID']])
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
    SWI = pd.merge(staysWishesAPI, instiPeso, left_on='Relation: ID',right_on='Relation: ID')
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
    

    t = t.dropna(axis=0,subset=['Stay wish ID'])
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
            #print('--- Entro a caso poco ---')
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
        revisar = 1
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
            try:
                xTrain,yTrain = oversample.fit_resample(xTrain,yTrain)
            except:
                print("error---------")
                print(u)
                revisar = 2
            if(revisar == 2):
                print("no toma esta uni")
            else:
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

def man(pCountry,pPrograma,pGpa,pPeso,pSeats,pLenguajes, ptipo):
    externalInstitutionsDF,stayOpportunityDF,relationInstitution,programasUniandes,staysAPI,staysWishesAPI,seatsDF,unisClase0,unisClase1,cupos,viajesE, staywishes = cargarData()
    SOI, SOI2 = dataPrepInic(externalInstitutionsDF,stayOpportunityDF,relationInstitution,programasUniandes)
    print("....soi......")
    print(SOI)
    print("...soi2.....")
    print(SOI2)
    print("-----------")
    filteredInstitutionsDF,gpa,peso,seats,filteredInstitutionsDF2 = validUniversitites(SOI,pCountry,pPrograma,pGpa,pPeso,pSeats,pLenguajes, SOI2)
    print("-----------")    
    print("-----------")    
    print("-----------")    
    print(filteredInstitutionsDF2)
    #print(filteredInstitutionsDF)
    t = prepDataForModel(filteredInstitutionsDF,staysAPI,staysWishesAPI,seatsDF,cupos)
    print("prep data")
    print(t)
    dataForModel = t[0]
    institutions = t[1]
    #print('--- Data For Modelo ---')
    #print(dataForModel)
    #print('--- Institutions ---')
    #print(institutions)
    
    universidadesPrediccion = KNN(dataForModel,institutions,viajesE,gpa,peso,seats,unisClase0,unisClase1)
    print("-----------")    
    print("-----------")    
    print("-----------")  
    print(universidadesPrediccion)
    F2, SMILE, CINDA = validarTipo(filteredInstitutionsDF2, universidadesPrediccion)
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
    print("a ver porfa")
    print(universidadesPrediccion)
    #print(f'Con un recall: {avg}')
    return r,contenedores, filteredInstitutionsDF2, universidadesPrediccion, staywishes, SMILE, CINDA, F2


def validarTipo(FIDF2, universidadesPrediccion):
    print("aaaaa")
    nombres = []
    for i in range(0, len(universidadesPrediccion)):
        nombres.append(list(universidadesPrediccion[i].keys())[0])
    print(len(nombres))
    print(FIDF2.columns)
    nuevoFID = FIDF2[(FIDF2["Institution"].isin(nombres)) &(FIDF2["Frameworks"].str.contains("Teaching") == False) & (FIDF2["Frameworks"].str.contains("Proyecto de Grado") == False) & (FIDF2["Frameworks"].str.contains("Double") == False)&(FIDF2["Frameworks"].str.contains("Faculty") == False)]
    
    #nuevo = nuevoFID[nuevoFID["Name"]=="Pontificia Universidad Católica de Chile"]

    #Separar por CINDA
    CINDAFID = nuevoFID[nuevoFID["Frameworks"].str.contains("CINDA")]
    
    #Separar por SMILE
    SMILDFID = nuevoFID[nuevoFID["Frameworks"].str.contains("SMILE")]
    
    #Los demas
    
    restFID = nuevoFID[(nuevoFID["Frameworks"].str.contains("SMILE") == False) & (nuevoFID["Frameworks"].str.contains("CINDA") == False) ]
    
    return restFID, SMILDFID, CINDAFID
    
def guardarDatos(programa, promedio, lenguajes, guardar):
    if guardar == []:
        guardarRetorno = ["dg.guarin20@uniandes.edu.co", programa, promedio, lenguajes, []]
    else:
        guardarRetorno = ["dg.guarin20@uniandes.edu.co", programa, promedio, lenguajes, guardar[4]]
    return guardarRetorno


def guardarUni(universidad, porcentaje, guardar, FIDF2, guardaCount):
    if(guardaCount==0):
        return guardar
    else:
        uni = FIDF2[(FIDF2["Institution"]==universidad) & (FIDF2["Frameworks"].str.contains("Exchange"))]
        uni2 = uni.drop_duplicates(subset=["Institution","Frameworks"], keep= "first")
        print("-----------------------------------------..")
        print(FIDF2)
        print(FIDF2["Institution"].unique())
        
        print("uniiiii")
        print(uni2)
        print("fdsafsdafsadfsad")
        print(guardar[4])
        unis = guardar[4]
        uni2 = uni2[['Frameworks','Country', 'City', 'Institution', 'Language requirement 1', 'Language requirement 2',  'Minimum GPA/4', 'Dificultad SO']]
        uni2['Porcentaje'] = porcentaje
        uni2["Minimum GPA/4"].fillna("0", inplace = True)
        uni2["Language requirement 1"].fillna("No language requirement", inplace = True)
        uni2["Language requirement 2"].fillna("No language requirement", inplace = True)
        unis.append(uni2)
        guardar[4]= unis
        
        
        return guardar

def alistarDatos(FIDF2, universidadesPrediccion):
    nombres = []
    promedio = []
    for i in range(0, len(universidadesPrediccion)):
        nombres.append(list(universidadesPrediccion[i].keys())[0])
    print(promedio)
    unis = FIDF2[(FIDF2["Institution"].isin(nombres)) &(FIDF2["Frameworks"].str.contains("Teaching") == False) & (FIDF2["Frameworks"].str.contains("Proyecto de Grado") == False) & (FIDF2["Frameworks"].str.contains("Double") == False)&(FIDF2["Frameworks"].str.contains("Faculty") == False)]
    unis = unis[['Frameworks','Institution: ID','Official Language','Country', 'City', 'Institution', 'Language requirement 1', 'Language requirement 2', 'Language cerf score 2', 'Minimum GPA/4', 'Dificultad SO']]
    a = []
    
    for i in range(0, len(unis)):
        name = unis.iloc[i]['Institution']
        promedio = 0
        for j in range(0, len(universidadesPrediccion)):
            valor = list(universidadesPrediccion[j].keys())[0]
            if(valor == name):
                promedio = universidadesPrediccion[j][valor]
                j = 10000
        a.append(promedio) 
    unis['Porcentaje'] = a
    
    junto = []
    for i in range(0, len(unis)):
        country = unis.iloc[i]['Country']
        city = unis.iloc[i]['City']
        name = unis.iloc[i]['Institution']
        language1 = unis.iloc[i]['Language requirement 1']
        language2 = unis.iloc[i]['Language requirement 2']
        gpa = unis.iloc[i]['Minimum GPA/4']
        oficialL = unis.iloc[i]['Official Language']
        gpa2 = str(gpa)
        dificultad = unis.iloc[i]['Dificultad SO']
        if(dificultad >= 0.75):
            dm = "Dificil"
        elif((dificultad < 0.75) & (dificultad >= 0.25) ):
            dm = "MedioDificil"
        else:
            dm = "Facil"
        porcentaje = unis.iloc[i]['Porcentaje']
        if( porcentaje>= 0.75):
            pm = "Alto"
        elif((porcentaje< 0.75) & (porcentaje>=0.25) ):
            pm = "Medio"
        else:
            pm = "Bajo"
            
        message = "Country: " + country + " City: "+ city + " Uni Name: "+ name +" Official Language: "+oficialL+ " Language requirement 1: " + language1 + " Language requirement 2: "+ language2 + " GPA: " + gpa2 + " Dificultad: " + dm + " Porcentaje: " + pm
        junto.append(message)
    unis['Mensaje'] = junto
    return unis


def get_recommendation(lista, cosine_sim, indices, unis):
    valores = []
    for i in range(0, len(lista)):
        l = lista.iloc[i]
        nombre = l['Institution']
        try:
            indx = indices[nombre]
            example = np.int64(40)

            if type(indx) == type(example):
                dictio = {nombre : indx}
                indice = pd.Series(dictio)
            else:
                indice = indx
            sim_score = enumerate(cosine_sim[indice][0])
            sim_score = sorted(sim_score, key= lambda x:x[1], reverse = True)
            sim_index = [l[0] for l in sim_score]
            valor = unis.iloc[sim_index]
    
            valores.append(valor)
        except:
            print("error")
    return valores

def definirElGuardado(guarda ):
    g = guarda[4]
    guardaN = []
    guardaR = []
    guardaP = []
    
    for i in range(0, len(g)):
        print("-----")
        print(g)
        print(g[i].columns)
        guardaN.append(g[i]['Institution'].values[0])
        guardaR.append(g[i]['Frameworks'].values[0])
        guardaP.append(g[i]['Porcentaje'].values[0])
    return guardaN, guardaR, guardaP

def organizar(recomendacion, guarda, guardaCount):
    if(guardaCount == 0):
        rr = recomendacion.drop_duplicates(subset=["Institution"], keep= "first")
        print(rr[:10])
        print(rr["Institution"].unique())
        print(rr["Institution"].head(5))
        l = rr[:10]
        inst = []
        rel = []
        por = []
        print("hola")
        for j in range(0, len(l)):
            inst.append(rr.iloc[j]["Institution"])
            por.append(rr.iloc[j]["Porcentaje"])
            rel.append(rr.iloc[j]["Frameworks"])
        df = pd.DataFrame()
        df['Nombre'] = inst
        df["Porcentaje"] = por
        df['relation'] = rel
        df2 = df.sort_values(by=['Porcentaje'], ascending=False)
        print("porfavorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
        print(df2)
        return df2
    else:
        lista = recomendacion[0][:10]
       
        for i in range(1, len(recomendacion)):
            r = recomendacion[i]
            if( i == 1):
                l = pd.concat([lista, r[:10]])
            else:
                l = pd.concat([l, r[:10]])
    
        duplicado = l.pivot_table(columns=['Institution','Frameworks'], aggfunc='size')
        llaves = duplicado.keys()
        a = []
        nombres = []
        relations = []
        porcentajes = []
        for i in range(0, len(llaves)):
            nombre, relation = llaves[i]
            porcentaje = l['Porcentaje'][(l['Institution']==nombre)  & (l['Frameworks']==relation)]

            porcentajes.append(porcentaje.values[0])
            nombres.append(nombre)
            relations.append(relation)
            a.append(duplicado[llaves[i]])
        
        df = pd.DataFrame()
        df['Duplicado'] = a
        df['Nombre'] = nombres
        df['relation'] = relations
        df['Porcentaje'] = porcentajes
        df2 = df.sort_values(by=['Duplicado'], ascending=False)
        g= guarda[4]
        if guardaCount == 1:
            print("no paso")
        else:
            for i in range(0, len(g)):
                name = g[i]['Institution'].values[0]
                df2 = df2[df2.Nombre != name]
            
        return df2
    
    
def probabilidad(porcentaje):
    contenedores = []
    for prob in porcentaje:
        if prob >= 0 and prob < 0.3:
            contenedores.append('Baja')
        elif prob >= 0.3 and prob < 0.7:
            contenedores.append('Media')
        else:
            contenedores.append('Alta')
    return contenedores   
    
def SistemRecomendacionContent(unis, guarda):
    td = TfidfVectorizer(stop_words="english")

#for i in range(0, len(columnas)):
 #   print(unis[columnas[i]].unique()) 

    unis["Minimum GPA/4"].fillna("0", inplace = True)
    unis["Language requirement 1"].fillna("No language requirement", inplace = True)
    unis["Language requirement 2"].fillna("No language requirement", inplace = True)
    unis["Frameworks"].drop
    td_matrix = td.fit_transform(unis['Mensaje'])
    print("matrix")
    print(td_matrix)
    cosine_sim = linear_kernel(td_matrix, td_matrix)
    print("cosine")
    print(cosine_sim)
    lista = guarda[4]
    ind = [x for x in range(0, len(unis))]
    indices = pd.Series(ind, index=unis['Institution']).drop_duplicates()
  
    recomendacion = get_recommendation(lista, cosine_sim, indices, unis)
    
    return recomendacion 
def SistemaRecomendacionCollab(staywishes, guarda, unis):

    wishes = staywishes[staywishes['Form'].str.contains("Outgoing", na=False)]
    wishes = wishes[['Institution','Person: ID','Status selection', 'Stay: Home - Degree programme']]

    personas = wishes['Person: ID'].unique()
    personInst = []
    for i in range(0, len(personas)):
        v = wishes[wishes['Person: ID']== personas[i]]
        institutos = v['Institution'].values
        status = v['Status selection'].values
        degree = v['Stay: Home - Degree programme'].unique()[0]
        if(len(institutos)>1):
            unir = [personas[i], institutos, status, degree, 0]
            personInst.append(unir)

    bachelor = guarda[1]
    institutosg = []
    print("guardaaaaa")
    print(guarda)
    
    for j in range(0, len(guarda[4])):
        institutosg.append(guarda[4][j]['Institution'].values[0])

    indexes = []
    for i in range(0, len(personInst)):
            for j in range(0, len(institutosg)):
                for k in range(0, len(personInst[i][1])):
                    if((institutosg[j] == personInst[i][1][k])):
                        indexes.append(i)
                    
    personDegree = []
    for i in range(0, len(indexes)):
        if(personInst[indexes[i]][3]==bachelor):
            personInst[indexes[i]][4] += 1 
        personDegree.append(personInst[indexes[i]])
        
    differentesUni = []
    for i in range(0, len(personDegree)):
        un = personDegree[i][1]
        for j in range(0, len(un)):
            if((un[j] in institutosg)== False):
                if((un[j] in differentesUni)==False):
                    differentesUni.append(un[j])
    print(differentesUni)
    personUni = []
    personUni2 = []
    for i in range(0, len(differentesUni)):
        suma = 0
        for j in range(0, len(personDegree)):
            if(differentesUni[i] in personDegree[j][1]):
                suma += 1
                suma += personDegree[j][4]
        personUni.append(differentesUni[i])
        personUni2.append(suma)
    df = pd.DataFrame()
    df['Uni'] = personUni
    df['valor'] = personUni2
    df2 = df.sort_values(by=['valor'], ascending=False)
    percent = []
    for i in range(0, len(df2)):
        universidad = unis[unis['Institution']==df2['Uni'][i]]
        if(len(universidad)==0):
            percent.append(0)
        else:
            p = universidad['Porcentaje'].values
            print(p)
            percent.append(p[0])

    df2['Porcentaje']=percent
    print(type(percent[0]))
    df3 = df2.sort_values(by=['Porcentaje'], ascending = False)
    return df3

def alistarDatos2(FIDF2, universidadesPrediccion, lenguaje):
    print("alistarDatos")
    print(FIDF2.columns)
    nombres = universidadesPrediccion["Uni"].to_numpy()
    porcentaje  = universidadesPrediccion["Porcentaje"].to_numpy()
    nuevoFID = FIDF2[(FIDF2["Institution"].isin(nombres)) &(FIDF2["Frameworks"].str.contains("Teaching") == False) & (FIDF2["Frameworks"].str.contains("Proyecto de Grado") == False) & (FIDF2["Frameworks"].str.contains("Double") == False)&(FIDF2["Frameworks"].str.contains("Faculty") == False)]
    unis = nuevoFID[(nuevoFID["Frameworks"].str.contains("Confusio") == False)  & (nuevoFID["Frameworks"].str.contains("Faculty") == False) & (nuevoFID["Frameworks"].str.contains("Staff") == False)& (nuevoFID["Frameworks"].str.contains("Master") == False) 
    & (nuevoFID["Frameworks"].str.contains("phd") == False) & (nuevoFID["Frameworks"].str.contains("Co-supervision") == False)
    & (nuevoFID["Frameworks"].str.contains("Co-tutelle") == False) & (nuevoFID["Frameworks"].str.contains("Administrative Staff") == False)
    & (nuevoFID["Frameworks"].str.contains("Freemover") == False) & (nuevoFID["Frameworks"].str.contains("Participation in events") == False)
    & (nuevoFID["Frameworks"].str.contains("Proyecto de Grado") == False) & (nuevoFID["Frameworks"].str.contains("Research Agreement") == False)
    & (nuevoFID["Frameworks"].str.contains("Scholarship") == False) & (nuevoFID["Frameworks"].str.contains("Sigueme") == False)
    &(nuevoFID["Frameworks"].str.contains("Teaching Assistant") == False) & (nuevoFID["Frameworks"].str.contains("Test-framework") == False)
    & (nuevoFID["Frameworks"].str.contains("Program Master") == False)]
    
    
    
    unis = unis[['Frameworks','Country', 'City','Official Language', 'Institution',  'Language requirement 1', 'Language requirement 2', 'Language cerf score 2', 'Minimum GPA/4', 'Dificultad SO']]
    a = []
    
    for i in range(0, len(unis)):
        name = unis.iloc[i]['Institution']
        for j in range(0, len(nombres)):
            if(nombres[j] == name):
                a.append(porcentaje[j])
            
         #   valor = list(universidadesPrediccion[j].keys())[0]
          #  if(valor == name):
           #     promedio = universidadesPrediccion[j][valor]
            #    j = 10000
        #a.append(promedio) 
    print(len(a))
    print(len(unis))
    unis['Porcentaje'] = a
    
    junto = []
    print("Columnasssss")
    print(unis.columns)
    for i in range(0, len(unis)):
        country = unis.iloc[i]['Country']
        city = unis.iloc[i]['City']
        name = unis.iloc[i]['Institution']
        language1 = unis.iloc[i]['Language requirement 1']
        language2 = unis.iloc[i]['Language requirement 2']
        gpa = unis.iloc[i]['Minimum GPA/4']
        oficialL = unis.iloc[i]['Official Language']
        gpa2 = str(gpa)
        dificultad = unis.iloc[i]['Dificultad SO']
        if(dificultad >= 0.75):
            dm = "Dificil"
        elif((dificultad < 0.75) & (dificultad >= 0.25) ):
            dm = "MedioDificil"
        else:
            dm = "Facil"
        porcentaje = unis.iloc[i]['Porcentaje']
        if( porcentaje>= 0.75):
            pm = "Alto"
        elif((porcentaje< 0.75) & (porcentaje>=0.25) ):
            pm = "Medio"
        else:
            pm = "Bajo"
       
        message = "Country: " + country + " City: "+ city + " Uni Name: "+ name +" Official Language: "+oficialL+ " Language requirement 1: " + language1 + " Language requirement 2: "+ language2 + " GPA: " + gpa2 + " Dificultad: " + dm + " Porcentaje: " + pm
        junto.append(message)
    unis['Mensaje'] = junto
    unisRetorno = unis.sort_values(by=["Porcentaje", "Institution"], ascending = False)
    uniRetorno = unisRetorno.drop_duplicates(subset=["Institution"], keep= "first")
    uniR = uniRetorno.drop(['Mensaje', 'Language cerf score 2'], axis=1)
    guarda = ["dg.guarin20@uniandes.edu.co", "Administration Bsc", "5", ["German","English"], uniR ]
    return guarda
print("")
def sacarTipo(unis, opciones):
    name = []
    for i in range(0, len(opciones)):
        print("----")
        nombre = opciones[i]
        uni = unis[unis["Institution"]== nombre]
        n = uni["Frameworks"].values[0].upper()
        nombres = n.split(nombre.upper())
        nombreRetorno = nombres[0].replace("-"," ")
        name.append(nombreRetorno)
        
    return name
def organizarRelation (n, o):
    tipos = []
    for i in range(0, len(o)):
        nombre = n[i]
        tipoO = o[i].upper()
        tipo = tipoO.split(nombre.upper())
        tipoRetorno = tipo[0].replace("-"," ")
        tipos.append(tipoRetorno)
    return tipos

def obtenerLink(unis, opciones):
    link = []
    for i in range(0, len(opciones)):
        nombre = opciones[i]
        name = unis[unis["Institution"]==nombre]
        print(name.columns)
        nID = name['Institution: ID'].values[0]
        url = "https://uniandes.moveonca.com/publisher/institution/1/"
        url2 = "/spa?relTypes=4&frmTypes=24|26|27|3|36&acadYears=&acadPeriods=&directions=2&defaultRelStatus=2&inst_int_settings_filter=2|4|5|6|7|8|9|10|11|12|13&acad_year_display=&acad_period_display=&document_types=1|5&restriction_types=1&restriction_id_filter=1&inst_document_types=1|5&inst_restriction_types=1&keyword=&country=10&institution_external=&degree_programme=&instance=2970&publisherId=1"
        nID2 = str(nID).split(".")[0]
        urlTotal = url + nID2 + url2
        link.append(urlTotal)
    return link

    