import cv2
import face_recognition as fr
from sklearn import neighbors
import pickle
import os
import numpy as np

# Caminho onde o modelo KNN será salvo ou carregado
model_path = "modelo_knn.clf"
n_neighbors = 3  # Número de vizinhos para o KNN
testePath = "teste.jpeg" # imagem para ser testada

def load_training_data(dataset_path):
    """
    Carrega as imagens dos colaboradores, extrai os encodings faciais e
    associa cada encoding ao papel correspondente.
    """
    encodings = []
    labels = []

    # Percorre cada pasta do dataset (cada pasta representa um papel)
    for papel in os.listdir(dataset_path):
        pasta_papel = os.path.join(dataset_path, papel)

        if not os.path.isdir(pasta_papel):  # Ignora arquivos que não são pastas
            continue

        # Itera sobre cada imagem na pasta do papel
        for imagem_file in os.listdir(pasta_papel):
            imagem_path = os.path.join(pasta_papel, imagem_file)
            imagem = fr.load_image_file(imagem_path)  # Carrega a imagem
            face_encodings = fr.face_encodings(imagem)  # Extrai o encoding

            # Verifica se a imagem possui pelo menos um rosto
            if len(face_encodings) > 0:
                encodings.append(face_encodings[0])  # Salva o encoding
                labels.append(papel)  # Salva o papel correspondente (classe)

    return encodings, labels

# Treina o modelo ou carrega o modelo existente
def train_or_load_model(dataset_path):

    #Treina o modelo KNN ou carrega um modelo existente.
    if not os.path.isfile(model_path):
        print("Treinando o modelo KNN...")
        encodings, labels = load_training_data(dataset_path)

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

def recognize_person_knn(image, knn_clf, distance_threshold=0.9):
    """
    # Usa o modelo KNN treinado para prever o papel de uma pessoa recebendo a
    imagem, o modelo KNN e o nivel de aceitação para o modelo.
    
    :param image: Imagem a ser analisada
    :param knn_clf: Modelo KNN treinado
    :param distance_threshold: Limiar de distância para considerar um rosto conhecido
    :return: Papel reconhecido ou 'Desconhecido'
    """
    # Codifica a imagem fornecida
    face_encodings = fr.face_encodings(image)

    # Verifica se a imagem contém pelo menos um rosto
    if len(face_encodings) == 0:
        return "Nenhum rosto encontrado na imagem."

    # Obtém o primeiro rosto da imagem
    face_encoding = face_encodings[0]
    # Calcula as distâncias para os vizinhos mais próximos
    closest_distances = knn_clf.kneighbors([face_encoding], n_neighbors=n_neighbors)
    # Verifica se a distância do vizinho mais próximo está dentro do limiar aceitável
    if closest_distances[0][0][0] > distance_threshold:
        return "Desconhecido"
    
    prediction = knn_clf.predict([face_encoding])
    return prediction[0]

# Exemplo de uso
def main():
    knn_clf = train_or_load_model("dataset/")

    # Carrega a imagem e converte para RGB
    img = fr.load_image_file(testePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = fr.face_locations(img)
    
    if len(face_locations) > 0:
        faceLoc = face_locations[0]
        
        # Reconhece a pessoa na imagem usando o modelo KNN
        papel = recognize_person_knn(img, knn_clf)
        
        # Desenha retângulo ao redor do rosto
        cv2.rectangle(img, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 255, 0), 2)
        
        # Exibe o papel reconhecido
        cv2.putText(img, papel, (faceLoc[3], faceLoc[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    cv2.imshow("Imagem", img)
    # Salva o papel em um arquivo de texto
    with open("output.txt", "w") as f:
        f.write(papel)
    # Executa o script de controle de acesso
    exec(open("cofreAps.py").read())

    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()