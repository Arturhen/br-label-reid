import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

# Largura desejada para as imagens
IMAGE_WIDTH = 300

# Função para carregar os dados do CSV
def carregar_dados(csv_file):
    return pd.read_csv(csv_file)

# Função para carregar as respostas do CSV de respostas
def carregar_respostas(csv_file):
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    else:
        return pd.DataFrame(columns=['Folder1', 'Folder2', 'Equal'])

# Função para criar a interface gráfica principal
def criar_interface(dados, pasta1, pasta2, pasta_csv):
    global respostas
    csv_respostas = os.path.join(pasta_csv, 'respostas.csv')
    respostas = carregar_respostas(csv_respostas)

    # Ordenar os dados pela coluna 'AverageDistance'
    dados = dados.sort_values(by='AverageDistance')

    # Função para avançar para a próxima imagem
    def proxima_imagem():
        nonlocal index
        global respostas

        while index < len(dados):
            folder1 = dados['Folder1'].iloc[index]
            folder2 = dados['Folder2'].iloc[index]

            if respostas[((respostas['Folder1'] == folder1) & (respostas['Folder2'] == folder2)) | 
                         ((respostas['Folder1'] == folder1) & (respostas['Equal'] == 'Sim')) | 
                         ((respostas['Folder2'] == folder2) & (respostas['Equal'] == 'Sim'))].empty:
                mostrar_imagens(index)
                break
            else:
                index += 1
                respostas = carregar_respostas(csv_respostas)

    # Função para mostrar as imagens atuais
    def mostrar_imagens(index):
        folder1_path = os.path.join(pasta1, str(dados['Folder1'].iloc[index]))
        folder2_path = os.path.join(pasta2, str(dados['Folder2'].iloc[index]))
  
        imagens1 = carregar_imagens(folder1_path)
        imagens2 = carregar_imagens(folder2_path)

        # Exibir imagens na lista esquerda
        for i, imagem in enumerate(imagens1):
            imagem = imagem.resize((IMAGE_WIDTH, int(imagem.size[1] * IMAGE_WIDTH / imagem.size[0])))
            imagem = ImageTk.PhotoImage(imagem)
            lista_imagens1[i].config(image=imagem)
            lista_imagens1[i].image = imagem

        # Exibir imagens na lista direita
        for i, imagem in enumerate(imagens2):
            imagem = imagem.resize((IMAGE_WIDTH, int(imagem.size[1] * IMAGE_WIDTH / imagem.size[0])))
            imagem = ImageTk.PhotoImage(imagem)
            lista_imagens2[i].config(image=imagem)
            lista_imagens2[i].image = imagem

        distancia_label.config(text=f"Distância Média: {dados['AverageDistance'].iloc[index]}")
        pasta1_label.config(text=f"Pasta 1: {dados['Folder1'].iloc[index]}")
        pasta2_label.config(text=f"Pasta 2: {dados['Folder2'].iloc[index]}")

    # Função para carregar todas as imagens de uma pasta
    def carregar_imagens(folder_path):
        imagens = []
        for file in os.listdir(folder_path):
            if file.endswith('.jpg'):
                imagem_path = os.path.join(folder_path, file)
                imagem = Image.open(imagem_path)
                imagens.append(imagem)
        return imagens

    # Função para salvar a resposta no CSV
    def salvar_resposta(resposta):
        arquivo = os.path.join(pasta_csv, 'respostas.csv')
        
        if not os.path.exists(arquivo):
            with open(arquivo, 'w') as f:
                f.write("Folder1,Folder2,Equal\n")
        
        with open(arquivo, 'a') as f:
            f.write(f"{dados['Folder1'].iloc[index]},{dados['Folder2'].iloc[index]},{resposta}\n")

    # Função chamada quando o botão "Sim" é clicado
    def sim_clicked():
        salvar_resposta('Sim')
        nonlocal index
        index += 1
        proxima_imagem()

    # Função chamada quando o botão "Não" é clicado
    def nao_clicked():
        salvar_resposta('Não')
        nonlocal index
        index += 1
        proxima_imagem()

    # Iniciar o índice na primeira imagem
    index = 0

    # Criando a janela principal
    root = tk.Tk()
    root.title("Identificação de Objetos")

    # Frame para as imagens da pasta 1
    frame1 = tk.Frame(root)
    frame1.grid(row=0, column=0, padx=10, pady=10)
    pasta1_label = tk.Label(frame1, text="", font=("Arial", 12))
    pasta1_label.pack()
    lista_imagens1 = []
    for _ in range(5):  # Ajuste conforme o número máximo de imagens esperadas
        label = tk.Label(frame1)
        label.pack(side=tk.LEFT)
        lista_imagens1.append(label)

    # Frame para as imagens da pasta 2
    frame2 = tk.Frame(root)
    frame2.grid(row=0, column=1, padx=10, pady=10)
    pasta2_label = tk.Label(frame2, text="", font=("Arial", 12))
    pasta2_label.pack()
    lista_imagens2 = []
    for _ in range(5):  # Ajuste conforme o número máximo de imagens esperadas
        label = tk.Label(frame2)
        label.pack(side=tk.LEFT)
        lista_imagens2.append(label)

    # Label para exibir a distância média
    distancia_label = tk.Label(root, text="", font=("Arial", 14))
    distancia_label.grid(row=1, columnspan=2, pady=10)

    # Botões para responder se é o mesmo objeto ou não
    button_font = ('Helvetica', 12)
    sim_button = tk.Button(root, text="Sim", width=20, height=5, font=button_font, bg="green", fg="white", command=sim_clicked)
    sim_button.grid(row=2, column=0, padx=10, pady=10)
    
    nao_button = tk.Button(root, text="Não", width=20, height=5, font=button_font, bg="red", fg="white", command=nao_clicked)
    nao_button.grid(row=2, column=1, padx=10, pady=10)

    # Mostrar a primeira imagem
    proxima_imagem()

    root.mainloop()

# Tela inicial para selecionar pastas e CSVs
def tela_inicial():
    def selecionar_pasta(label):
        pasta_selecionada = filedialog.askdirectory(title="Selecione a Pasta")
        label.config(text=pasta_selecionada)

    def iniciar():
        pasta_entrada_1 = label_pasta_entrada_1.cget("text")
        pasta_entrada_2 = label_pasta_entrada_2.cget("text")
        pasta_csv_dir = entry_csv_dir.get()

        if not pasta_entrada_1 or not pasta_entrada_2 or not pasta_csv_dir:
            messagebox.showerror("Erro", "Por favor, selecione todas as pastas e insira um diretório válido.")
            return

        csv_file_path = filedialog.askopenfilename(
            title="Selecione o arquivo CSV de similaridade",
            filetypes=[("Arquivos CSV", "*.csv")]
        )

        if not csv_file_path:
            messagebox.showerror("Erro", "Por favor, selecione um arquivo CSV válido.")
            return

        # Criar o diretório para salvar os CSVs, se não existir
        os.makedirs(pasta_csv_dir, exist_ok=True)

        # Fechar a janela inicial e iniciar a interface principal
        root.destroy()

        dados_csv = carregar_dados(csv_file_path)
        criar_interface(dados_csv, pasta_entrada_1, pasta_entrada_2, pasta_csv_dir)

    def show_help():
        help_text = (
            "1. Selecione as pastas de entrada e o diretório para salvar os CSVs.\n"
            "2. Selecione o arquivo CSV de similaridade.\n"
            "3. Clique em 'Iniciar' para começar a conferência das imagens.\n"
            "4. Use os botões 'Sim' e 'Não' para indicar se as imagens mostradas são do mesmo objeto.\n"
            "5. As respostas serão salvas automaticamente no arquivo 'respostas.csv'."
        )
        messagebox.showinfo("Ajuda", help_text)

    # Criar a janela inicial
    root = tk.Tk()
    root.title("Configuração Inicial")
    root.geometry("800x400")

    # Frame para os inputs
    frame_inputs = tk.Frame(root)
    frame_inputs.pack(pady=20)

    # Seleção da primeira pasta
    tk.Label(frame_inputs, text="Pasta 1:", font=("Arial", 12)).grid(row=0, column=0, padx=10, pady=10)
    label_pasta_entrada_1 = tk.Label(frame_inputs, text="", font=("Arial", 12), bg="#f0f0f0", width=40, anchor="w")
    label_pasta_entrada_1.grid(row=0, column=1, padx=10, pady=10)
    tk.Button(frame_inputs, text="Selecionar", command=lambda: selecionar_pasta(label_pasta_entrada_1)).grid(row=0, column=2, padx=10, pady=10)

    # Seleção da segunda pasta
    tk.Label(frame_inputs, text="Pasta 2:", font=("Arial", 12)).grid(row=1, column=0, padx=10, pady=10)
    label_pasta_entrada_2 = tk.Label(frame_inputs, text="", font=("Arial", 12), bg="#f0f0f0", width=40, anchor="w")
    label_pasta_entrada_2.grid(row=1, column=1, padx=10, pady=10)
    tk.Button(frame_inputs, text="Selecionar", command=lambda: selecionar_pasta(label_pasta_entrada_2)).grid(row=1, column=2, padx=10, pady=10)

    # Campo para inserir o diretório de saída dos CSVs
    tk.Label(frame_inputs, text="Diretório para salvar os CSVs:", font=("Arial", 12)).grid(row=2, column=0, padx=10, pady=10)
    entry_csv_dir = tk.Entry(frame_inputs, font=("Arial", 12), width=40)
    entry_csv_dir.grid(row=2, column=1, padx=10, pady=10)

    # Botão para iniciar o processamento
    tk.Button(root, text="Iniciar", font=("Arial", 14), bg="#4CAF50", fg="white", command=iniciar).pack(pady=20)

    # Botão de ajuda
    help_button = tk.Button(root, text="Ajuda", command=show_help)
    help_button.pack(pady=10)

    root.mainloop()


# Caminho para o arquivo CSV de respostas (será gerado na pasta selecionada)
csv_respostas = './respostas.csv'

# Iniciar a tela inicial
tela_inicial()
