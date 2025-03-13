import cv2
import threading
from tkinter import Tk, Label, Entry, Button, Frame, filedialog, messagebox, ttk

def select_roi(frame):
    # Seletor de ROI com OpenCV
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    return roi

def cut_video(input_video_path, output_video_path, progress_bar):
    try:
        # Abrindo o vídeo de entrada
        cap = cv2.VideoCapture(input_video_path)
        
        if not cap.isOpened():
            messagebox.showerror("Erro", "Erro ao abrir o vídeo.")
            return

        # Lendo o primeiro frame
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Erro", "Erro ao ler o vídeo.")
            cap.release()
            return

        # Selecionando a ROI no primeiro frame
        roi = select_roi(frame)
        x, y, w, h = roi

        # Definindo o codec e criando o objeto VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Cortando o frame com base na ROI
            cropped_frame = frame[y:y+h, x:x+w]
            
            # Escrevendo o frame cortado no vídeo de saída
            out.write(cropped_frame)

            # Atualizando a barra de progresso
            processed_frames += 1
            progress_percentage = (processed_frames / total_frames) * 100
            progress_bar["value"] = progress_percentage
            progress_bar.update()

        cap.release()
        out.release()
        messagebox.showinfo("Sucesso", "Vídeo cortado e salvo com sucesso!")
        root.quit()  # Fechar o programa após o processamento
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro: {e}")

def select_input_file(entry_input):
    # Abrir diálogo para selecionar arquivo de entrada e preencher no campo de texto
    file_path = filedialog.askopenfilename(
        title="Selecione o vídeo de entrada",
        filetypes=[("Arquivos de vídeo", "*.avi *.mp4 *.mkv"), ("Todos os arquivos", "*.*")]
    
    )
    entry_input.delete(0, "end")
    entry_input.insert(0, file_path)

def start_processing(entry_input, entry_output, progress_bar):
    input_video_path = entry_input.get()
    output_video_path = entry_output.get()

    if not input_video_path or not output_video_path:
        messagebox.showwarning("Aviso", "Por favor, preencha todos os campos.")
        return
    
    # Iniciar processamento em uma thread separada para evitar travamento da interface
    threading.Thread(target=cut_video, args=(input_video_path, output_video_path, progress_bar)).start()

def show_help():
    help_text = (
        "1. Selecione o vídeo de entrada clicando no botão 'Selecionar'.\n"
        "2. Digite o nome do arquivo de saída.\n"
        "3. Clique em 'Iniciar Processamento' para começar a cortar o vídeo.\n"
        "4. Uma janela será aberta para selecionar a região de interesse (ROI) no primeiro frame do vídeo, para selecionar clique segure e arraste, após aperte ENTER para começar.\n"
        "5. O vídeo será processado e salvo com a ROI selecionada."
    )
    messagebox.showinfo("Ajuda", help_text)

def main():
    global root  # Tornar a variável root global para ser acessada em outras funções
    root = Tk()
    root.title("LABEL BR REID - Editor de Vídeo")
    root.geometry("600x400")
    
    # Estilo geral da interface
    root.configure(bg="#f0f0f0")

    # Título do programa
    label_title = Label(root, text="LABEL BR REID - Editor de Vídeo", font=("Arial", 20), bg="#f0f0f0", fg="#333")
    label_title.pack(pady=20)

    # Campo para selecionar ou digitar o caminho do vídeo de entrada
    frame_input = Frame(root, bg="#f0f0f0")
    frame_input.pack(pady=10)

    label_input = Label(frame_input, text="Vídeo de Entrada:", font=("Arial", 14), bg="#f0f0f0", fg="#333")
    label_input.pack(side="left")

    entry_input = Entry(frame_input, width=40)
    entry_input.pack(side="left", padx=10)

    button_browse_input = Button(frame_input, text="Selecionar", command=lambda: select_input_file(entry_input))
    button_browse_input.pack(side="left")

    # Campo para digitar o nome do arquivo de saída
    frame_output = Frame(root, bg="#f0f0f0")
    frame_output.pack(pady=10)

    label_output = Label(frame_output, text="Nome do Arquivo de Saída:", font=("Arial", 14), bg="#f0f0f0", fg="#333")
    label_output.pack(side="left")

    entry_output = Entry(frame_output, width=40)
    entry_output.pack(side="left", padx=10)

    # Aviso para incluir a extensão do arquivo
    label_extension_warning = Label(root, text="Não esqueça de incluir a extensão .mp4, .avi ou .mkv", font=("Arial", 10), bg="#f0f0f0", fg="red")
    label_extension_warning.pack(pady=5)

    # Barra de progresso moderna
    progress_bar_label = Label(root, text="Progresso:", font=("Arial", 12), bg="#f0f0f0", fg="#333")
    progress_bar_label.pack(pady=10)

    progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
    
    button_start = Button(root, text="Iniciar Processamento", font=("Arial", 12), bg="#4CAF50", fg="white",
                          command=lambda: start_processing(entry_input, entry_output, progress_bar))
    
    button_start.pack(pady=20)
    
    progress_bar.pack(pady=10)

    # Botão de ajuda
    help_button = Button(root, text="Ajuda", command=show_help)
    help_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
