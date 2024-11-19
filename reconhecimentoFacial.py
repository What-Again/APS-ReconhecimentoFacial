import cv2  # Importa a biblioteca OpenCV para manipulação de imagens
import face_recognition as fr  # Importa a biblioteca face_recognition para reconhecimento facial
from sklearn import neighbors  # Importa o KNeighbors para classificação
import pickle
import os

# Caminho onde o modelo KNN será salvo ou carregado
model_path = "modelo_knn.clf"
n_neighbors = 3  # Número de vizinhos para o KNN


# Função para carregar dados de treinamento e treinar o modelo KNN
def load_training_data(dataset_path):
    """
    Carrega as imagens dos colaboradores, extrai os encodings faciais e
    associa cada encoding ao papel correspondente (ex: Gerente, Administradora).
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

    return encodings, labels  # Retorna os encodings e labels para treinamento


# Verifica se o modelo já foi treinado anteriormente
if not os.path.isfile(model_path):
    # Treina o modelo se ele não estiver salvo
    print("Treinando o modelo KNN...")
    encodings, labels = load_training_data("dataset/")  # Carrega os dados de treinamento

    # Inicializa o classificador KNN e o treina com os encodings e labels
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', weights='distance')
    knn_clf.fit(encodings, labels)  # Treina o modelo com os dados de treinamento

    # Salva o modelo treinado para uso posterior
    with open(model_path, 'wb') as f:
        pickle.dump(knn_clf, f)
    print("Modelo KNN treinado e salvo com sucesso!")
else:
    print("Modelo KNN já existente. Carregando o modelo...")


# Função para reconhecer uma nova pessoa usando o modelo treinado
def recognize_person_knn(image, model_path):
    """
    Usa o modelo KNN treinado para prever o papel de uma pessoa
    em uma imagem fornecida.
    """
    # Carrega o modelo KNN salvo
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    # Codifica a imagem fornecida
    face_encodings = fr.face_encodings(image)

    # Verifica se a imagem contém pelo menos um rosto
    if len(face_encodings) == 0:
        return "Nenhum rosto encontrado na imagem."

    # Usa o modelo para prever o papel do rosto na imagem
    prediction = knn_clf.predict([face_encodings[0]])

    return prediction[0]


# Carrega a imagem e converte para RGB
img = fr.load_image_file('teste.jpg')  # Imagem para ser testada
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detecta a localização do rosto
faceLoc = fr.face_locations(img)[0]  # Encontra a localização da face e pega a primeira detecção
cv2.rectangle(img, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 255, 0), 2)

# Reconhece a pessoa na imagem usando o modelo KNN
papel = recognize_person_knn(img, model_path)
print(f"Papel reconhecido: {papel}")

# Exibe a imagem com o papel reconhecido sobreposto
cv2.putText(img, papel, (faceLoc[3], faceLoc[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow("Imagem", img)

# Salva o papel reconhecido em um arquivo de texto e executa o cofreAps.py
with open("output.txt", "w") as f:
    f.write(papel)
# Nivel de permisão da pessoa analisada.
exec(open("cofreAps.py").read())

# Aguarda uma tecla para fechar as janelas
cv2.waitKey(0)
cv2.destroyAllWindows()
