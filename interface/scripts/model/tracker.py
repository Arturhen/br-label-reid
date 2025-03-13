from ultralytics import YOLO
import cv2
import os
import numpy as np
from .SORT import Sort
import torch
import time
import csv
from tqdm import tqdm  # Importar tqdm para a barra de progresso

def process_video(video_source, output_dir, fps=300):
    # Inicializar o modelo YOLOv8
    model = YOLO('yolov8n.pt')  # Certifique-se de que o caminho do modelo está correto

    # Definir o dispositivo (CPU ou GPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Carregar o vídeo
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_source}.")
        return

    # Inicializar o rastreador SORT
    tracker = Sort(iou_threshold=0.15)

    # Criar diretório para salvar as imagens se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Contar o número total de frames no vídeo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Criar uma barra de progresso
    progress_bar = tqdm(total=total_frames, desc="Processando Frames", unit="frame")

    # Dicionário para armazenar os arquivos CSV dos objetos
    csv_files = {}
    last_seen = {}

    frame_count = 0
    prev_time = time.time()
    frame_interval = 1.0 / fps  # Intervalo de tempo entre frames

    while True:
        current_time = time.time()
        elapsed_time = current_time - prev_time
        if elapsed_time < frame_interval:
            # Calcular o tempo restante para aguardar
            sleep_time = frame_interval - elapsed_time
            time.sleep(sleep_time)
            # Atualizar o tempo da última leitura
        prev_time = current_time + sleep_time

        # Ler o próximo frame do vídeo
        ret, frame = cap.read()
        if not ret:
            break

        # Realizar a detecção de objetos no frame
        results = model(frame, classes=2, conf=0.35, verbose=False)  # Desativar verbose

        # Processar cada resultado
        boxes = []
        if results:  # Verifica se há resultados
            for result in results:
                # Acessar as caixas delimitadoras
                for box in result.boxes:
                    # Cada box contém coordenadas e outras informações
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    boxes.append([x1, y1, x2, y2, conf])

        boxes = np.array(boxes)

        # Rastrear os objetos usando o SORT
        tracks = tracker.update(boxes)

        # Obter o timestamp atual
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        # Loop através dos tracks
        for track in tracks:
            x1, y1, x2, y2, track_id = track.astype(np.int32)
            # Verificar se as coordenadas estão dentro dos limites da imagem
            if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                if x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                    # Nome do arquivo com timestamp
                    filename = f"{track_id}_{timestamp}_{frame_count}.jpg"
                    # Caminho completo do arquivo
                    filepath = os.path.join(output_dir, str(track_id), filename)
                    # Criar o diretório para o track_id se não existir
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    # Salvar a imagem
                    cv2.imwrite(filepath, frame[y1:y2, x1:x2])

                    # Adicionar dados ao arquivo CSV do objeto
                    track_id_str = str(track_id)
                    if track_id_str not in csv_files:
                        csv_path = os.path.join(output_dir, track_id_str, 'tracking_info.csv')
                        csv_files[track_id_str] = open(csv_path, mode='w', newline='')
                        csv_writer = csv.writer(csv_files[track_id_str])
                        # Escrever cabeçalhos no CSV
                        csv_writer.writerow(['id', 'bounding_box', 'numero_do_frame', 'nome_do_video'])
                    else:
                        csv_writer = csv.writer(csv_files[track_id_str])

                    # Escrever dados no arquivo CSV
                    bounding_box = f"{x1},{y1},{x2},{y2}"
                    csv_writer.writerow([track_id, bounding_box, frame_count, os.path.basename(video_source)])

                    # Atualizar o último tempo visto para o track_id
                    last_seen[track_id_str] = current_time

        # Fechar arquivos CSV que não foram vistos nos últimos 10 segundos
        for track_id_str in list(last_seen.keys()):
            if current_time - last_seen[track_id_str] > 10:
                csv_files[track_id_str].close()
                del csv_files[track_id_str]
                del last_seen[track_id_str]

        frame_count += 1
        progress_bar.update(1)  # Atualizar a barra de progresso

    cap.release()
    progress_bar.close()  # Fechar a barra de progresso

    # Fechar todos os arquivos CSV abertos
    for file in csv_files.values():
        file.close()

    print(f"Processamento concluído para {video_source}. Imagens e CSVs salvos em: {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Processar e rastrear vídeos com YOLOv8 e SORT.')
    parser.add_argument('video_source', type=str, help='Caminho para o vídeo ou URL RTSP')
    parser.add_argument('output_dir', type=str, help='Diretório para salvar as imagens e CSVs rastreados')

    args = parser.parse_args()
    process_video(args.video_source, args.output_dir)
