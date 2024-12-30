import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

class StyledApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EmotiAI")
        self.root.geometry("1280x720")
        self.root.configure(bg="#E7DDFF")  # Softer background color
        self.current_screen = None

        # Custom style
        self.style = ttk.Style()
        self.configure_styles()

        # Call pantalla_bienvenida to start the app
        self.pantalla_bienvenida()

    def configure_styles(self):
        """Configure custom styles for widgets"""
        # Configure colors
        self.primary_color = "#FFFFFF"  # White for buttons
        self.secondary_color = "#000000"  # Black text
        self.text_color = "#1F2937"  # Dark gray

        # Button style
        self.style.configure("TButton", 
            font=("Segoe UI", 12), 
            background=self.primary_color,
            foreground=self.secondary_color,
            padding=(15, 10),
            borderwidth=1,
            relief="solid"
        )
        self.style.map("TButton",
            background=[('active', "#F0F0F0")],  # Light gray on hover
        )

        # Entry style
        self.style.configure("TEntry", 
            font=("Segoe UI", 12), 
            padding=(10, 5)
        )

        # Label style
        self.style.configure("TLabel", 
            font=("Segoe UI", 12), 
            background="#F0F4F8",
            foreground=self.text_color
        )

    def create_rounded_button(self, parent, text, command):
        """Create a styled button with rounded corners"""
        button = ttk.Button(parent, text=text, command=command, style="TButton")
        return button

    def pantalla_bienvenida(self):
        """Improved welcome screen"""
        self.ocultarPantalla()

        frame1 = tk.Frame(self.root, bg="#F0F4F8")
        frame1.pack(fill="both", expand=True)

        # Title with modern styling
        label1 = ttk.Label(frame1, text="EmotiAI", style="TLabel", font=("Segoe UI", 36, "bold"))
        label1.pack(pady=30)

        # Description with improved typography
        label2 = ttk.Label(
            frame1,
            text=(
                "EmotiAI entiende el lenguaje de tus emociones en el mundo digital.\n"
                "A través de inteligencia artificial empática, transformamos tus\n"
                "publicaciones en conexiones significativas."
            ),
            style="TLabel",
            font=("Segoe UI", 16),
            anchor="center"
        )
        label2.pack(padx=20, pady=20)

        # Load and resize image with a frame for better presentation
        image_frame = tk.Frame(frame1, bg="#F0F4F8", borderwidth=2, relief="solid")
        image_frame.pack(pady=20)

        try:
            image = Image.open("caritas.jpg")
            image = image.resize((250, 250))
            photo = ImageTk.PhotoImage(image)
            image_label = tk.Label(image_frame, image=photo, bg="#F0F4F8")
            image_label.photo = photo
            image_label.pack(padx=10, pady=10)
        except Exception as e:
            print(f"Error al cargar la imagen: {e}")
            image_label = ttk.Label(image_frame, text="Imagen no encontrada", style="TLabel")
            image_label.pack(padx=10, pady=10)

        # Styled start button
        button = self.create_rounded_button(frame1, "Comenzar", self.menu)
        button.pack(pady=30)

        self.current_screen = frame1

    def menu(self):
        """Improved main menu"""
        self.ocultarPantalla()

        frame2 = tk.Frame(self.root, bg="#F0F4F8")
        frame2.pack(fill="both", expand=True)

        labeltitulo = ttk.Label(frame2, text="¿Qué te gustaría hacer?", style="TLabel", font=("Segoe UI", 36, "bold"))
        labeltitulo.pack(pady=50)

        # Centered, styled button
        combined_button = self.create_rounded_button(
            frame2, 
            "Análisis de Imagen y Texto", 
            self.pantallaAnalisis
        )
        combined_button.pack(pady=30)

        self.current_screen = frame2

    def pantallaAnalisis(self):
        """Improved analysis screen with better layout and styling"""
        self.ocultarPantalla()

        frame = tk.Frame(self.root, bg="#F0F4F8")
        frame.pack(fill="both", expand=True)

        # Title with modern styling
        labeltitulo = ttk.Label(frame, text="Análisis de Imágenes y Texto", style="TLabel", font=("Segoe UI", 36, "bold"))
        labeltitulo.pack(pady=30)

        # Image Analysis Section
        image_frame = tk.Frame(frame, bg="#F0F4F8")
        image_frame.pack(pady=20)

        label_imagen = ttk.Label(image_frame, text="Análisis de Imágenes", style="TLabel", font=("Segoe UI", 24))
        label_imagen.pack(pady=10)

        upload_button = self.create_rounded_button(image_frame, "Subir Archivo", self.subir_archivo)
        upload_button.pack(pady=10)

        self.image_result_label = ttk.Label(image_frame, text="Resultado de la imagen: ---", style="TLabel")
        self.image_result_label.pack(pady=10)

        # Text Analysis Section
        text_frame = tk.Frame(frame, bg="#F0F4F8")
        text_frame.pack(pady=20)

        label_texto = ttk.Label(text_frame, text="Análisis de Texto", style="TLabel", font=("Segoe UI", 24))
        label_texto.pack(pady=10)

        # Improved input styling
        input_frame = tk.Frame(text_frame, bg="#F0F4F8", borderwidth=1, relief="solid")
        input_frame.pack(pady=10)
        inputTexto = ttk.Entry(input_frame, width=40, style="TEntry")
        inputTexto.pack(padx=5, pady=5)

        analyze_text_button = self.create_rounded_button(
            text_frame,
            "Analizar Texto",
            lambda: self.mostrar_resultado_texto(inputTexto.get())
        )
        analyze_text_button.pack(pady=10)

        self.text_result_label = ttk.Label(text_frame, text="Resultado del texto: ---", style="TLabel")
        self.text_result_label.pack(pady=10)

        # Combined Analysis Section
        combined_analyze_button = self.create_rounded_button(
            text_frame,
            "Analizar Imagen y Texto",
            lambda: self.analizar_imagen_y_texto(inputTexto.get())
        )
        combined_analyze_button.pack(pady=10)

        self.combined_result_label = ttk.Label(
            text_frame, 
            text="Resultado del análisis combinado: ---", 
            style="TLabel", 
            wraplength=800
        )
        self.combined_result_label.pack(pady=10)

        # Return to menu button
        boton_regresar = self.create_rounded_button(frame, "Regresar al Menú", self.menu)
        boton_regresar.pack(pady=20)

        self.current_screen = frame

    def mostrar_resultado_texto(self, texto):
        """Show text analysis result"""
        resultado = f"Tú estás 70% feliz basado en: {texto}"
        self.text_result_label.config(text=resultado)

    def subir_archivo(self):
        """Upload and display image file"""
        archivo = filedialog.askopenfilename(title="Seleccionar archivo", filetypes=[("Archivos de imagen", ".jpg;.jpeg;*.png")])
        if archivo:
            print(f"Archivo seleccionado: {archivo}")
            try:
                image = Image.open(archivo)
                image = image.resize((250, 250))
                photo = ImageTk.PhotoImage(image)
                self.image_result_label.config(image=photo, text="")
                self.image_result_label.photo = photo
            except Exception as e:
                print(f"Error al cargar la imagen: {e}")
                self.image_result_label.config(text="Error al cargar la imagen")

    def analizar_imagen_y_texto(self, texto):
        """Simulate combined image and text analysis"""
        resultado = f"Análisis combinado:\nTexto analizado: {texto}\nImagen cargada: Procesada correctamente."
        self.combined_result_label.config(text=resultado)

    def ocultarPantalla(self):
        """Hide current screen"""
        if self.current_screen:
            self.current_screen.pack_forget()

# Create main window
root = tk.Tk()
app = StyledApp(root)
root.mainloop()