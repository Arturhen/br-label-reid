import shutil
import tkinter as tk
from tkinter import  filedialog, ttk, messagebox
import threading
import os
from model.tracker import process_video
import concurrent.futures
from PIL import Image


def process_sources(sources, output_dir, keep_only_three_images):
    # Função para processar cada source com seu ID e pasta de saída
    def process_source(source, camera_id, output_subdir):
        try:
            print(f"Processando: {source} | ID da Câmera: {camera_id} | Saída: {output_subdir}")
            process_video(source, output_subdir)
            print(f"Concluído: {source}")
        except Exception as e:
            print(f"Erro ao processar {source}: {e}")

    # Usar ThreadPoolExecutor para processar todas as fontes em paralelo
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sources), 4)) as executor:
        futures = []

        for source, camera_id in sources.items():
            output_subdir = os.path.join(output_dir, f"c{camera_id}")
            os.makedirs(output_subdir, exist_ok=True)
            futures.append(executor.submit(process_source, source, camera_id, output_subdir))
        
        # Aguardar a conclusão de todas as tarefas
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Ocorreu um erro: {exc}")

    if keep_only_three_images:
        for camera_id in sources.values():
            output_subdir = os.path.join(output_dir, f"c{camera_id}")
            process_folder(output_subdir)

    messagebox.showinfo("Sucesso", "Todos os vídeos foram processados!")

def process_folder(folder):
    def list_files(folder):
        return sorted(os.listdir(folder))

    def remove_small_images(folder):
        files = list_files(folder)
        jpg_files = [file for file in files if file.lower().endswith('.jpg')]
        
        for file in jpg_files:
            file_path = os.path.join(folder, file)
            with Image.open(file_path) as img:
                if img.height < 100:
                    os.remove(file_path)
                    print(f"Imagem {file} removida (menos de 100 pixels de altura).")

    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)
        if os.path.isdir(subdir_path):
            remove_small_images(subdir_path)
            
            files = list_files(subdir_path)
            jpg_files = [file for file in files if file.lower().endswith('.jpg')]
            num_jpg_files = len(jpg_files)
            
            if num_jpg_files < 3:
                shutil.rmtree(subdir_path)
                print(f"Pasta {subdir_path} apagada (menos de 3 imagens .jpg).")
                continue
            
            first_index = 0
            middle_index = num_jpg_files // 2
            last_index = num_jpg_files - 1
            
            selected_images = [jpg_files[first_index], jpg_files[middle_index], jpg_files[last_index]]
            
            for file in jpg_files:
                if file not in selected_images:
                    os.remove(os.path.join(subdir_path, file))
            
            print(f"Pasta {subdir_path} processada.")

def start_processing(sources, output_dir_entry, keep_only_three_images_var):
    output_dir = output_dir_entry.get()
    if not output_dir:
        messagebox.showwarning("Aviso", "Por favor, insira o diretório de saída.")
        return

    if not sources:
        messagebox.showwarning("Aviso", "Nenhum source foi adicionado.")
        return

    keep_only_three_images = keep_only_three_images_var.get()

    # Iniciar processamento em uma thread separada
    threading.Thread(target=process_sources, args=(sources, output_dir, keep_only_three_images)).start()

def add_source(sources_listbox, sources):
    # Abrir diálogo para selecionar arquivos ou URLs RTSP
    file_path = filedialog.askopenfilename(
        title="Selecione um vídeo ou insira uma URL RTSP",
        filetypes=[("Arquivos de vídeo", "*.avi *.mp4 *.mkv"), ("Todos os arquivos", "*.*")]
    )
    
    if file_path:
        # Pedir o ID da câmera associado ao source
        camera_id = tk.simpledialog.askstring("ID da Câmera", "Insira o ID da câmera:")
        if not camera_id:
            messagebox.showwarning("Aviso", "Você deve inserir um ID para a câmera.")
            return

        # Adicionar à lista de sources e atualizar a exibição na interface
        sources[file_path] = camera_id
        sources_listbox.insert(tk.END, f"Source: {file_path} | Camera ID: {camera_id}")

def show_help():
    help_text = (
        "1. Clique em 'Adicionar Source' para adicionar vídeos ou URLs RTSP.\n"
        "2. Insira o ID da câmera para cada source adicionado.\n"
        "3. Selecione o diretório base de saída.\n"
        "4. Clique em 'Iniciar Processamento' para começar a processar os vídeos.\n"
        "5. A barra de progresso mostrará o andamento do processamento."
    )
    messagebox.showinfo("Ajuda", help_text)

def main():
    # Configuração principal da janela Tkinter
    root = tk.Tk()
    root.title("Processador de Vídeos - YOLOv8 + SORT")
    root.geometry("700x600")

    # Dicionário para armazenar os sources e IDs das câmeras
    sources = {}

    # Título do programa
    label_title = tk.Label(root, text="Processador de Vídeos - YOLOv8 + SORT", font=("Arial", 16))
    label_title.pack(pady=10)

    # Botão para adicionar novos sources
    add_source_button = tk.Button(root, text="Adicionar Source", font=("Arial", 12),
                                   command=lambda: add_source(sources_listbox, sources))
    add_source_button.pack(pady=10)

    # Listbox para exibir os sources adicionados
    sources_listbox = tk.Listbox(root, width=80, height=10)
    sources_listbox.pack(pady=10)

    # Campo para definir o diretório base de saída
    frame_output_dir = tk.Frame(root)
    frame_output_dir.pack(pady=10)

    label_output_dir = tk.Label(frame_output_dir, text="Diretório Base de Saída:", font=("Arial", 12))
    label_output_dir.pack(side=tk.LEFT)

    output_dir_entry = tk.Entry(frame_output_dir, width=40)
    output_dir_entry.pack(side=tk.LEFT, padx=10)

    browse_output_button = tk.Button(frame_output_dir, text="Selecionar",
                                      command=lambda: output_dir_entry.insert(0,
                                                                               filedialog.askdirectory(
                                                                                   title="Selecione o Diretório Base de Saída")))
    browse_output_button.pack(side=tk.LEFT)

    # Checkbox para manter apenas 3 imagens por ID
    keep_only_three_images_var = tk.BooleanVar()
    keep_only_three_images_check = tk.Checkbutton(root, text="Manter apenas 3 imagens por ID", variable=keep_only_three_images_var)
    keep_only_three_images_check.pack(pady=10)

    # Botão para iniciar o processamento
    start_button = tk.Button(root, text="Iniciar Processamento", font=("Arial", 12), bg="#4CAF50", fg="white",
                              command=lambda: start_processing(sources, output_dir_entry, keep_only_three_images_var))
    start_button.pack(pady=20)

    # Botão de ajuda
    help_button = tk.Button(root, text="Ajuda", command=show_help)
    help_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
