# APS Reconhecimento Facial

## Descrição
Projeto Universitario para reconhecimento facial desenvolvido em python.

## Dependencias Utilizadas.
- Python 3.10*
- pip
- dlib 19.22.99
- CMAKE
- face-recognition
- numpy
- opencv-python
- scikit-learn

### Instalar Dependências
```bash
pip install -r requirements.txt
```

## Uso
Colocar um arquivo "teste.jpg"(imagem a ser analisada) dentro da pasta, caso o arquivo seja de outra extensão alterar na variavel testePath reconhecimentoFacial.py. Antes de rodar o programa pela primeira vez deletar modelo_knn.clf por conta da possibilidade de conflito, um novo modelo sera treinado ao inicializar reconhecimentoFacial.py.

## Como rodar
Iniciar reconhecimentoFacial.py.
