import cv2
import face_recognition as fr
from sklearn import neighbors
import pickle
import os
import numpy as np

# Caminho onde o modelo KNN será salvo ou carregado
model_path = "modelo_knn.clf"
n_neighbors = 3

def carregar_modelo(dataset_path):
    # Carrega as imagens dos, extrai os rostos e associa cada rosto a um papel
    encodings = []
    labels = []
    # Percorre cada pasta do dataset sendo cada pasta um papel)
    for papel in os.listdir(dataset_path):
        pasta_papel = os.path.join(dataset_path, papel)

        if not os.path.isdir(pasta_papel):
            continue

        for imagem_file in os.listdir(pasta_papel):
            imagem_path = os.path.join(pasta_papel, imagem_file)
            imagem = fr.load_image_file(imagem_path)
            face_encodings = fr.face_encodings(imagem)
            # Verifica se a imagem possui pelo menos um rosto
            if len(face_encodings) > 0:
                encodings.append(face_encodings[0])  
                labels.append(papel)
                print(f"Loaded face for {papel} from {imagem_file}")

    print(f"Total faces loaded: {len(encodings)}")
    print(f"Unique labels: {set(labels)}")
    return encodings, labels

# Treina o modelo ou carrega o modelo existente
def treinar_modelo(dataset_path):
    if not os.path.isfile(model_path):
        print("Treinando o modelo KNN...")
        encodings, labels = carregar_modelo(dataset_path)
        # Inicializa o classificador KNN e o treina com os encodings e labels
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', weights='distance')
        knn_clf.fit(encodings, labels)

        with open(model_path, 'wb') as f:
            pickle.dump(knn_clf, f)
        print("Modelo KNN treinado e salvo com sucesso!")
        return knn_clf
    else:
        print("Modelo KNN já existente. Carregando o modelo...")
        with open(model_path, 'rb') as f:
            return pickle.load(f)

def define_pessoa(image, knn_clf, distance_threshold=0.4):
    # Usa o modelo KNN treinado para definir o papel de uma pessoa recebendo a
    #imagem, o modelo KNN e o nivel de aceitação para o modelo.
    face_encodings = fr.face_encodings(image)

    if len(face_encodings) == 0:
        print("Nenhum rosto encontrado na imagem.")
        return "Nenhum rosto encontrado"

    face_encoding = face_encodings[0]
    closest_distances = knn_clf.kneighbors([face_encoding], n_neighbors=n_neighbors)
    
    print(f"Closest distances: {closest_distances[0][0]}")
    print(f"Distance threshold: {distance_threshold}")

    if closest_distances[0][0][0] > distance_threshold:
        print("Acima do limiar de distância. Considerado desconhecido.")
        return "Desconhecido"
    
    prediction = knn_clf.predict([face_encoding])
    print(f"Predicted label: {prediction[0]}")
    return prediction[0]

def main():
    knn_clf = treinar_modelo("dataset/")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        rgb_frame = frame[:, :, ::-1]
        
        face_locations = fr.face_locations(rgb_frame)
        
        if len(face_locations) > 0:
            faceLoc = face_locations[0]
            
            papel = define_pessoa(rgb_frame, knn_clf)
            
            cv2.rectangle(frame, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 255, 0), 2)
            
            cv2.putText(frame, papel, (faceLoc[3], faceLoc[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow("Reconhecimento Facial", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
main()
