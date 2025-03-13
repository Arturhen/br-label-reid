import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
import csv
import random
from pathlib import Path

# Function to combine images classified as the same car from different cameras into a single folder
def create_pasta_conjunta(folder_1_path, folder_2_path, csv_file, output_base):
    if not os.path.exists(output_base):
        os.makedirs(output_base)

    index = 0
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder1 = row['Folder1'].strip()
            folder2 = row['Folder2'].strip()
            equal = row['Equal'].strip()

            if equal == 'Sim':
                new_folder_path = os.path.join(output_base, str(index))
                os.makedirs(new_folder_path)
                shutil.copytree(os.path.join(folder_1_path, folder1), os.path.join(new_folder_path, 'c1'))
                shutil.copytree(os.path.join(folder_2_path, folder2), os.path.join(new_folder_path, 'c2'))
                index += 1

# Function to rename images according to the camera
def rename_images(base_folder):
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path) and folder.isdigit():
            folder_num = folder
            folder_1_content = os.path.join(folder_path, 'c1')
            folder_2_content = os.path.join(folder_path, 'c2')
            camera_1_name = os.path.basename(folder_1_content.rstrip(os.sep))
            camera_2_name = os.path.basename(folder_2_content.rstrip(os.sep))

            if os.path.exists(folder_1_content):
                for img in os.listdir(folder_1_content):
                    if img.endswith('.jpg'):
                        new_name = f"{folder_num}_{camera_1_name}_{img.split('_', 1)[1]}"
                        shutil.move(os.path.join(folder_1_content, img), os.path.join(folder_path, new_name))
                shutil.rmtree(folder_1_content)

            if os.path.exists(folder_2_content):
                for img in os.listdir(folder_2_content):
                    if img.endswith('.jpg'):
                        new_name = f"{folder_num}_{camera_2_name}_{img.split('_', 1)[1]}"
                        shutil.move(os.path.join(folder_2_content, img), os.path.join(folder_path, new_name))
                shutil.rmtree(folder_2_content)

# Function to split folders into training and test sets
def split_train_test(diretorio_principal, pasta_treinamento, pasta_teste):
    Path(pasta_treinamento).mkdir(parents=True, exist_ok=True)
    Path(pasta_teste).mkdir(parents=True, exist_ok=True)

    def copiar_para_destino(pasta_origem, destino):
        arquivos = os.listdir(pasta_origem)
        pasta_destino = os.path.join(destino, os.path.basename(pasta_origem))
        Path(pasta_destino).mkdir(parents=True, exist_ok=True)
        for arquivo in arquivos:
            caminho_origem = os.path.join(pasta_origem, arquivo)
            caminho_destino = os.path.join(pasta_destino, arquivo)
            if os.path.isfile(caminho_origem):
                shutil.copy2(caminho_origem, caminho_destino)

    for pasta in os.listdir(diretorio_principal):
        caminho_pasta = os.path.join(diretorio_principal, pasta)
        if os.path.isdir(caminho_pasta):
            if random.random() < 0.75:
                copiar_para_destino(caminho_pasta, pasta_treinamento)
            else:
                copiar_para_destino(caminho_pasta, pasta_teste)

# Function to remove images from within the folders
def move_images_and_delete_folders(folder_path):
    if not os.path.exists(folder_path):
        return

    image_extensions = ['.jpg', '.jpeg', '.png']
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(folder_path, file)
                shutil.move(src_path, dst_path)

    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in dirs:
            folder_to_delete = os.path.join(root, name)
            if not os.listdir(folder_to_delete):
                os.rmdir(folder_to_delete)

# Function to split the test set into test and query sets
def split_test_query(source_folder, output_folder_1, output_folder_2):
    os.makedirs(output_folder_1, exist_ok=True)
    os.makedirs(output_folder_2, exist_ok=True)

    images_by_id = {}
    image_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    for image_file in image_files:
        if image_file.endswith('.jpg'):
            id_uuid = image_file.split('_')[0]
            if id_uuid not in images_by_id:
                images_by_id[id_uuid] = []
            images_by_id[id_uuid].append(image_file)

    for id_uuid, images in images_by_id.items():
        images.sort()
        if len(images) >= 6:
            half = len(images) // 2
            first_half = images[:half]
            second_half = images[half:]
            if random.random() < 0.5:
                selected_images = first_half[:3]
            else:
                selected_images = second_half[-3:]
            for image in selected_images:
                src_path = os.path.join(source_folder, image)
                dst_path = os.path.join(output_folder_1, image)
                shutil.move(src_path, dst_path)
            for image in images:
                if image not in selected_images:
                    src_path = os.path.join(source_folder, image)
                    dst_path = os.path.join(output_folder_2, image)
                    shutil.move(src_path, dst_path)

# Function to clean up intermediate folders
def clean_up_intermediate_folders(base_folder):
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path) and folder not in ['image_train', 'image_test', 'image_query']:
            shutil.rmtree(folder_path)

# Function to create the user interface
def create_interface():
    def selecionar_pasta(label):
        pasta_selecionada = filedialog.askdirectory(title="Selecione a Pasta")
        label.config(text=pasta_selecionada)

    def selecionar_arquivo(label):
        arquivo_selecionado = filedialog.askopenfilename(title="Selecione o Arquivo CSV", filetypes=[("Arquivos CSV", "*.csv")])
        label.config(text=arquivo_selecionado)

    def iniciar():
        pasta_entrada_1 = label_pasta_entrada_1.cget("text")
        pasta_entrada_2 = label_pasta_entrada_2.cget("text")
        csv_file_path = label_csv_file.cget("text")
        pasta_saida = label_pasta_saida.cget("text")

        if not pasta_entrada_1 or not pasta_entrada_2 or not csv_file_path or not pasta_saida:
            messagebox.showerror("Erro", "Por favor, selecione todas as pastas e o arquivo CSV.")
            return

        # Step 1: Combine images classified as the same car from different cameras into a single folder
        create_pasta_conjunta(pasta_entrada_1, pasta_entrada_2, csv_file_path, pasta_saida)

        # Step 2: Rename images according to the camera
        rename_images(pasta_saida)

        # Step 3: Split folders into training and test sets
        pasta_treinamento = os.path.join(pasta_saida, 'image_train')
        pasta_teste = os.path.join(pasta_saida, 'image_test')
        split_train_test(pasta_saida, pasta_treinamento, pasta_teste)

        # Step 4: Remove images from within the folders
        move_images_and_delete_folders(pasta_treinamento)
        move_images_and_delete_folders(pasta_teste)

        # Step 5: Split the test set into test and query sets
        pasta_query = os.path.join(pasta_saida, 'image_query')
        split_test_query(pasta_teste, pasta_teste, pasta_query)

        # Step 6: Clean up intermediate folders
        clean_up_intermediate_folders(pasta_saida)

        messagebox.showinfo("Sucesso", "Processamento concluído!")

    def show_help():
        help_text = (
            "1. Selecione as pastas referentes as pastas 1 e 2 do csv resposta. \n"
            "2. Selecione o arquivo CSV de resposta.\n"
            "3. Selecione a pasta de saída.\n"
            "4. Clique em 'Iniciar' para começar o processamento do dataset.\n"
            "5. As imagens serão combinadas, renomeadas e divididas em conjuntos de treinamento e teste.\n"
            "6. As imagens de teste serão divididas em conjuntos de teste e consulta.\n"
            "7. Você terá um dataset pronto para ser utilizado em um modelo de reidentificação de objetos."
        )
        messagebox.showinfo("Ajuda", help_text)

    # Create the main window
    root = tk.Tk()
    root.title("Processamento de Dataset")
    root.geometry("800x400")

    # Frame for inputs
    frame_inputs = tk.Frame(root)
    frame_inputs.pack(pady=20)

    # Select the first folder
    tk.Label(frame_inputs, text="Pasta 1:", font=("Arial", 12)).grid(row=0, column=0, padx=10, pady=10)
    label_pasta_entrada_1 = tk.Label(frame_inputs, text="", font=("Arial", 12), bg="#f0f0f0", width=40, anchor="w")
    label_pasta_entrada_1.grid(row=0, column=1, padx=10, pady=10)
    tk.Button(frame_inputs, text="Selecionar", command=lambda: selecionar_pasta(label_pasta_entrada_1)).grid(row=0, column=2, padx=10, pady=10)

    # Select the second folder
    tk.Label(frame_inputs, text="Pasta 2:", font=("Arial", 12)).grid(row=1, column=0, padx=10, pady=10)
    label_pasta_entrada_2 = tk.Label(frame_inputs, text="", font=("Arial", 12), bg="#f0f0f0", width=40, anchor="w")
    label_pasta_entrada_2.grid(row=1, column=1, padx=10, pady=10)
    tk.Button(frame_inputs, text="Selecionar", command=lambda: selecionar_pasta(label_pasta_entrada_2)).grid(row=1, column=2, padx=10, pady=10)

    # Select the CSV file
    tk.Label(frame_inputs, text="Arquivo CSV de respostas (respostas.csv):", font=("Arial", 12)).grid(row=2, column=0, padx=10, pady=10)
    label_csv_file = tk.Label(frame_inputs, text="", font=("Arial", 12), bg="#f0f0f0", width=40, anchor="w")
    label_csv_file.grid(row=2, column=1, padx=10, pady=10)
    tk.Button(frame_inputs, text="Selecionar", command=lambda: selecionar_arquivo(label_csv_file)).grid(row=2, column=2, padx=10, pady=10)

    # Select the output folder
    tk.Label(frame_inputs, text="Pasta de Saída:", font=("Arial", 12)).grid(row=3, column=0, padx=10, pady=10)
    label_pasta_saida = tk.Label(frame_inputs, text="", font=("Arial", 12), bg="#f0f0f0", width=40, anchor="w")
    label_pasta_saida.grid(row=3, column=1, padx=10, pady=10)
    tk.Button(frame_inputs, text="Selecionar", command=lambda: selecionar_pasta(label_pasta_saida)).grid(row=3, column=2, padx=10, pady=10)

    # Button to start processing
    tk.Button(root, text="Iniciar", font=("Arial", 14), bg="#4CAF50", fg="white", command=iniciar).pack(pady=20)

    # Help button
    help_button = tk.Button(root, text="Ajuda", command=show_help)
    help_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_interface()
