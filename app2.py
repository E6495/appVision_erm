#Versión Final
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from tkinter import Tk, filedialog
from sklearn.metrics import classification_report, accuracy_score



data = pd.read_csv('C:/Users/elili/Downloads/proyectoVision/proyectoVision/base_datos_orb1.csv')
X = data.drop(columns=['clase'])
y = data['clase']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
clf = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=0)
clf.fit(X_train, y_train)
joblib.dump(clf, "modeloBaseAcercado8.pkl")  

y_pred = clf.predict(X_test)


print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))
print("Precisión del modelo:", accuracy_score(y_test, y_pred))


def predict_image(img, model):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is not None:
        descriptors = descriptors.flatten()
        if len(descriptors) > X.shape[1]:
            descriptors = descriptors[:X.shape[1]]
        else:
            descriptors = np.pad(descriptors, (0, X.shape[1] - len(descriptors)), 'constant')

        
        descriptors_df = pd.DataFrame([descriptors], columns=X.columns)

        prediction = model.predict(descriptors_df)
        
        
        x_coords = [int(kp.pt[0]) for kp in keypoints]
        y_coords = [int(kp.pt[1]) for kp in keypoints]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        return prediction[0], (x_min, y_min, x_max, y_max)
    else:
        return "No se encontraron características en la imagen.", None



def select_image_and_predict():
    Tk().withdraw()  
    image_path = filedialog.askopenfilename(title="Selecciona una imagen", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    
    if image_path:
        img = cv2.imread(image_path)
        prediction, bbox = predict_image(img, modelo_rf)
        
        if prediction != "No se encontraron características en la imagen.":
            
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img, f'Predicted: {prediction}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        
        cv2.imshow('Resultado', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


modelo_rf = joblib.load("modeloBaseAcercado.pkl")
select_image_and_predict()
