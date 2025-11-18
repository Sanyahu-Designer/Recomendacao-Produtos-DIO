import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import requests
from io import BytesIO
import os

# Configurações iniciais
print("Bibliotecas importadas com sucesso!")

# Carregar o modelo ResNet50 pré-treinado (sem a camada de classificação)
def load_model():
    model = ResNet50(weights='imagenet', 
                    include_top=False, 
                    pooling='avg', 
                    input_shape=(224, 224, 3))
    return model

model = load_model()
print("Modelo ResNet50 carregado com sucesso!")

def extract_features(img_path, model):
    """
    Extrai features de uma imagem usando o modelo ResNet50
    """
    # Carregar e redimensionar a imagem
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Converter para array numpy
    img_array = image.img_to_array(img)
    
    # Expandir dimensões para batch size 1
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Pré-processar a imagem
    img_preprocessed = preprocess_input(img_batch)
    
    # Extrair features
    features = model.predict(img_preprocessed)
    
    return features.flatten()

# Função para carregar múltiplas imagens
def load_images_from_folder(folder_path, model):
    """
    Carrega todas as imagens de uma pasta e extrai suas features
    """
    features_list = []
    image_paths = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            try:
                features = extract_features(img_path, model)
                features_list.append(features)
                image_paths.append(img_path)
                print(f"Imagem processada: {filename}")
            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")
    
    return np.array(features_list), image_paths

# Vamos criar uma pasta de exemplo com 4 imagens de produtos
def create_sample_images():
    """
    Cria imagens de exemplo para demonstração
    """
    # URLs de imagens de exemplo (produtos)
    sample_urls = [
        "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=300",  # Tênis
        "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=300",  # Relógio
        "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=300",  # Fones
        "https://images.unsplash.com/photo-1485955900006-10f4d324d411?w=300"   # Camiseta
    ]
    
    # Criar pasta para imagens
    os.makedirs('produtos', exist_ok=True)
    
    # Baixar e salvar imagens
    for i, url in enumerate(sample_urls):
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img_path = f'produtos/produto_{i+1}.jpg'
            img.save(img_path)
            print(f"Imagem {i+1} salva: {img_path}")
        except Exception as e:
            print(f"Erro ao baixar imagem {i+1}: {e}")
    
    return 'produtos'

# Criar imagens de exemplo
folder_path = create_sample_images()

# Extrair features de todas as imagens
features, image_paths = load_images_from_folder(folder_path, model)

print(f"Total de imagens processadas: {len(image_paths)}")
print(f"Shape das features: {features.shape}")

# Criar modelo KNN para encontrar similaridades
knn = NearestNeighbors(n_neighbors=4, metric='cosine')
knn.fit(features)

print("Modelo KNN treinado com sucesso!")

def recommend_similar_products(query_image_path, model, knn_model, image_paths, top_k=3):
    """
    Recomenda produtos similares baseado na imagem de consulta
    """
    # Extrair features da imagem de consulta
    query_features = extract_features(query_image_path, model)
    
    # Encontrar produtos similares
    distances, indices = knn_model.kneighbors([query_features])
    
    # Exibir resultados
    plt.figure(figsize=(15, 5))
    
    # Imagem original
    plt.subplot(1, top_k + 1, 1)
    original_img = mpimg.imread(query_image_path)
    plt.imshow(original_img)
    plt.title('Produto Consultado')
    plt.axis('off')
    
    # Produtos recomendados
    for i, (idx, dist) in enumerate(zip(indices[0][1:top_k+1], distances[0][1:top_k+1])):
        plt.subplot(1, top_k + 1, i + 2)
        similar_img = mpimg.imread(image_paths[idx])
        plt.imshow(similar_img)
        plt.title(f'Similar {i+1}\nDist: {dist:.3f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return indices[0][1:top_k+1], distances[0][1:top_k+1]

# Testar com a primeira imagem como exemplo
query_image = image_paths[0]  # Primeira imagem como consulta

print(f"Consultando similaridades para: {query_image}")

# Obter recomendações
recommended_indices, distances = recommend_similar_products(
    query_image, model, knn, image_paths
)

# Mostrar informações detalhadas
print("\n=== RECOMENDAÇÕES ===")
for i, (idx, dist) in enumerate(zip(recommended_indices, distances)):
    print(f"Recomendação {i+1}:")
    print(f"  - Imagem: {image_paths[idx]}")
    print(f"  - Similaridade: {1-dist:.3f}")
    print()

def interactive_recommendation():
    """
    Interface interativa para testar diferentes produtos
    """
    print("=== SISTEMA DE RECOMENDAÇÃO DE PRODUTOS ===")
    print("Produtos disponíveis:")
    
    for i, path in enumerate(image_paths):
        product_name = os.path.basename(path).split('.')[0]
        print(f"{i+1}. {product_name}")
    
    try:
        choice = int(input("\nEscolha um produto (1-4) para ver recomendações: ")) - 1
        
        if 0 <= choice < len(image_paths):
            query_img = image_paths[choice]
            print(f"\nRecomendações baseadas em: {os.path.basename(query_img)}")
            
            recommended_indices, distances = recommend_similar_products(
                query_img, model, knn, image_paths
            )
            
        else:
            print("Escolha inválida!")
            
    except ValueError:
        print("Por favor, digite um número válido!")

# Executar interface interativa
interactive_recommendation()

# Código compacto para entrega
class ProductRecommender:
    def __init__(self):
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        self.knn = None
        self.image_paths = []
        self.features = []
    
    def train(self, images_folder):
        """Treina o sistema com imagens da pasta"""
        self.features, self.image_paths = load_images_from_folder(images_folder, self.model)
        self.knn = NearestNeighbors(n_neighbors=4, metric='cosine')
        self.knn.fit(self.features)
        print(f"Sistema treinado com {len(self.image_paths)} produtos!")
    
    def recommend(self, query_image_path, top_k=3):
        """Recomenda produtos similares"""
        query_features = extract_features(query_image_path, self.model)
        distances, indices = self.knn.kneighbors([query_features])
        
        # Mostrar resultados
        self.display_recommendations(query_image_path, indices[0][1:top_k+1], distances[0][1:top_k+1])
        
        return indices[0][1:top_k+1]

    def display_recommendations(self, query_path, indices, distances):
        """Exibe as recomendações visualmente"""
        plt.figure(figsize=(15, 5))
        
        # Produto consultado
        plt.subplot(1, len(indices) + 1, 1)
        plt.imshow(mpimg.imread(query_path))
        plt.title('Produto Consultado')
        plt.axis('off')
        
        # Recomendações
        for i, (idx, dist) in enumerate(zip(indices, distances)):
            plt.subplot(1, len(indices) + 1, i + 2)
            plt.imshow(mpimg.imread(self.image_paths[idx]))
            plt.title(f'Recomendação {i+1}\nSimilaridade: {1-dist:.3f}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Uso do sistema
if __name__ == "__main__":
    recommender = ProductRecommender()
    recommender.train('produtos')
    
    # Testar com o primeiro produto
    recommender.recommend(recommender.image_paths[0])
