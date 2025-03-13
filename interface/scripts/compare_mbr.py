import os
import torch
from itertools import combinations
import pandas as pd
from tqdm import tqdm
import pickle
import tempfile
import shutil
from lib_MBR import FeatureExtractor

# Função para calcular a distância euclidiana entre dois tensores
def euclidean_distance(x, y):
    return torch.norm(x - y, dim=-1).item()

# Função para extrair características das imagens em uma pasta e salvar em um arquivo temporário
def extract_and_save_features(folder_path, extractor):
    features = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            feature = extractor(img_path)
            if len(feature) > 0:
                features.append(feature[0].cpu().numpy())  # Convertendo para numpy array para salvar
            
    return torch.tensor(features) if features else None

# Criando uma instância do FeatureExtractor
extractor = FeatureExtractor(
                config_path='/home/artur/Documents/br-label-reid/model/config.yaml',#mudar
                weights_path='/home/artur/Documents/br-label-reid/model/best_mAP.pt',#mudar
                # device= # Opcional
)

# Caminhos das pastas
detect_track = '/home/artur/Documents/br-label-reid/saida_ex/c001'#mudar
detect_track2 = '/home/artur/Documents/br-label-reid/saida_ex/c002'#mudar

# Criando pastas temporárias
temp_dir1 = tempfile.mkdtemp()
temp_dir2 = tempfile.mkdtemp()

try:
    # Dicionário para armazenar as distâncias
    distances = []

    # Listar todas as combinações de pastas para calcular o progresso
    all_combinations = [(folder1, folder2) for folder1 in os.listdir(detect_track) for folder2 in os.listdir(detect_track2)]

    # Percorrendo as pastas e calculando as distâncias com barra de progresso
    for folder1, folder2 in tqdm(all_combinations, desc="Calculating distances", unit="pair"):
        folder1_path = os.path.join(detect_track, folder1)
        folder1_features_file = os.path.join(temp_dir1, f'features_{folder1}.pkl')
        # Verificar se já existem características extraídas para folder1
        if not os.path.exists(folder1_features_file):
            features1 = extract_and_save_features(folder1_path, extractor)
            if features1 is None:
                continue
            with open(folder1_features_file, 'wb') as f:
                pickle.dump(features1.numpy(), f)
        else:
            features1 = torch.tensor(pickle.load(open(folder1_features_file, 'rb')))
        
        folder2_path = os.path.join(detect_track2, folder2)
        folder2_features_file = os.path.join(temp_dir2, f'features_{folder2}.pkl')
        
        # Verificar se já existem características extraídas para folder2
        if not os.path.exists(folder2_features_file):
            features2 = extract_and_save_features(folder2_path, extractor)
            if features2 is None:
                continue
            with open(folder2_features_file, 'wb') as f:
                pickle.dump(features2.numpy(), f)
        else:
            features2 = torch.tensor(pickle.load(open(folder2_features_file, 'rb')))
        
        # Calculando a média das distâncias euclidianas entre todas as combinações de características
        total_distance = 0
        count = 0
        for feat1 in features1:
            for feat2 in features2:
                total_distance += euclidean_distance(feat1, feat2)
                count += 1
        average_distance = total_distance / count
        distances.append((folder1, folder2, average_distance))

    # Salvando os resultados finais em um arquivo CSV
    df = pd.DataFrame(distances, columns=['Folder1', 'Folder2', 'AverageDistance'])
    df.to_csv('similarity_results.csv', index=False)

    # Salvando os resultados finais em um arquivo TXT
    with open('similarity_results.txt', 'w') as f:
        for folder1, folder2, avg_distance in distances:
            f.write(f'Folder {folder1} in detect_track is similar to Folder {folder2} in detect_track2 with an average distance of {avg_distance:.4f}\n')

    print('Results saved to similarity_results.csv and similarity_results.txt')

finally:
    # Removendo pastas temporárias
    shutil.rmtree(temp_dir1)
    shutil.rmtree(temp_dir2)
