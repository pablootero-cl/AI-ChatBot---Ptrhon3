# Por defecto he configurado  LLAMA3 de 8b, puedes re-configurarlo con cualquier AI de la lista
# (2.9GB)Activar GPU, Requiere tener instalado CUDA https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
# (204MB) Instalar ollama desde https://ollama.com de lo contrario  este python no funcinara.
# (4.7GB) Ejecutar terminal: "ollama run llama3"  esto instalara llama3 y lo correra.
# El modelo IA se guarda en C:\Users\XXXX\models
# Instalar librerias ollama y angchain_community.llms
#   pip install ollama, angchain_community.llms
# Requerimientos minimos para IA de 8B CPU i5 de 6 nucleos.
# Uso de RAM con IA de 8B:
# * Procesamiento de texto: 2-4 GB
# * Generación de textos: 4-6 GB
# * Conversaciones con usuarios: 8-12 GB
# Para mas detalles mirar tabla a continuacion
"""
Model 	      Parameters 	Size 	Download
Llama 3 	       8B 	    4.7GB 	ollama run llama3
Llama 3 	      70B        40GB 	ollama run llama3:70b
Phi 3 Mini  	 3.8B 	    2.3GB 	ollama run phi3
Phi 3 Medium 	  14B 	    7.9GB 	ollama run phi3:medium
Gemma 	           2B 	    1.4GB 	ollama run gemma:2b
Gemma 	           7B 	    4.8GB 	ollama run gemma:7b
Mistral 	       7B 	    4.1GB 	ollama run mistral
Moondream 2 	 1.4B 	    829MB 	ollama run moondream
Neural Chat 	   7B 	    4.1GB 	ollama run neural-chat
Starling 	       7B 	    4.1GB 	ollama run starling-lm
Code Llama         7B 	    3.8GB 	ollama run codellama
Llama 2 Uncensored 7B 	    3.8GB 	ollama run llama2-uncensored
LLaVA 	           7B 	    4.5GB 	ollama run llava
Solar 	        10.7B 	    6.1GB 	ollama run solar
"""

import ollama
from langchain_community.llms import Ollama
import sys
from os import system, name # Limpiar pantalla
import torch # Para GPU
import datetime # Fecha y hora
from colorama import Fore, Back, Style # Colores

# Puedes usar: llama3, llama3:70b, phi3, phi3:medium, gemma:2b, gemma:7b, mistral, moondream, neural-chat, starling-lm, codellama, llama2-uncensored, llava, solar
#--------------------------
MOTOR_IA = "llama3" # MODIFICALO SI QUIERES USAR OTRA IA
#--------------------------
"""
NOTA: 
CPU: Se recomienda una CPU moderna con al menos 8 núcleos para manejar operaciones en el backend y preprocesamiento de datos de manera eficiente.
GPU: Para el entrenamiento e inferencia del modelo, especialmente con el modelo de 70B, es crucial tener una o más GPU potentes
RAM: Mínimo 16 GB para el modelo de 8B y 32 GB o más para el modelo de 70B.
llama3 es 8b con un peso de 4.7GB aprox ()
llama3:70b tiene un peso de 40GB aprox 

"""

if MOTOR_IA == "llama3":
	IA_M = "Llama 3"
elif MOTOR_IA == "llama3:70b":
	IA_M = "Llama 3"
elif MOTOR_IA == "phi3":
	IA_M = "Phi 3 Mini"
elif MOTOR_IA == "phi3:medium":
	IA_M = "Phi 3 Medium"
elif MOTOR_IA == "gemma:2b":
	IA_M = "Gemma"
elif MOTOR_IA == "gemma:7b":
	IA_M = "Gemma"
elif MOTOR_IA == "mistral":
	IA_M = "Mistral"	
if MOTOR_IA == "moondream":
	IA_M = "Moondream 2"
elif MOTOR_IA == "neural-chat":
	IA_M = "Neural Chat"
elif MOTOR_IA == "starling-lm":
	IA_M = "Starling"
elif MOTOR_IA == "codellama":
	IA_M = "Code Llama"
elif MOTOR_IA == "llama2-uncensored":
	IA_M = "Llama 2 UNC."
elif MOTOR_IA == "llava":
	IA_M = "LlaVa"
elif MOTOR_IA == "solar":
	IA_M = "Solar"
	
def fecha():
	global formatted_date
	# Obtener la fecha actuales...
	now = datetime.datetime.now()
	# Formatear la fecha y hora para mostrarla en el formato deseado...
	formatted_date = now.strftime("%d/%m/%Y")

def hora():
	global formatted_time
	# Obtener la fecha y hora actuales...
	now = datetime.datetime.now()
	#Formatear la fecha y hora para mostrarla en el formato deseado...
	formatted_time = now.strftime("%H:%M ")

# Activa la GPU...
def gpu():
	global IA
	hora()
	fecha()
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Activa la GPU
	if device == "cuda:0":
		print(formatted_time + Style.BRIGHT + Fore.CYAN + IA_M + Style.RESET_ALL + Fore.RESET + ": " + formatted_date)
		IA = Style.BRIGHT + Fore.CYAN + IA_M + Style.RESET_ALL + Fore.RESET + ": "	
		hora()	
		print(formatted_time + IA + "GPU Activada...\n")
	else:		
		print(formatted_time + Style.BRIGHT + Fore.RED + IA_M + Style.RESET_ALL + Fore.RESET + ": "+ formatted_date)
		IA = Style.BRIGHT + Fore.RED + IA_M + Style.RESET_ALL + Fore.RESET + ": "
		hora()
		print(formatted_time + IA + "GPU: Desactivada ...")
		print(formatted_time + IA + Back.RED + Fore.WHITE + "Al no poder activar el uso de la GPU, las respuestas pueden ser mas lentas. ..." + Back.RESET + Fore.RESET)
		print(formatted_time + IA + "(" + Fore.GREEN + "Solucion" + Fore.RESET+ "): Instala CUDA desde su sitio official.\n" + Back.YELLOW + Fore.BLUE + "https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local" + Fore.RESET + Back.RESET)
		print(formatted_time + IA + "Nota: Cuda podria tener un tamaño de 2.9GB.\n")


def clear(): # Limpia pantalla.
	if name == 'nt':
		_ = system('cls')
	else:
		_ = system('clear')

def main():
    llm = Ollama(model=MOTOR_IA, temperature=0.7) # Llama al modelo IA elegido, temperatura, asigna nivel de creatividad

    while True:
        input_user = input(Fore.GREEN + " ---> User: " + Fore.RESET) # Almacena la pregunta

        if input_user.lower() in ["salir", "Salir", "SALIR", "exit", "Exit", "EXIT", "bye", "Bye", "BYE"]: # CIerra el programa
            hora()
            print(formatted_time + IA + "Adiós!")
            break

        respuesta = llm.invoke(input_user) # Genera la respuesta
        hora()
        print(formatted_time + IA + f"{respuesta}\n\n") # Notifica la respuesta
        
if __name__ == "__main__":
    clear() # limpia pantalla
    gpu()   # Intenta activar GPU
    hora()  # Captura la hora
    print(formatted_time + IA + "Nota: (Escribe '" + Back.RED + Fore.WHITE + "salir" + Back.RESET + Fore.RESET + "' para terminar.): ")    
    print(formatted_time + IA + "¿En que te puedo ayudar?: ")
    main()  # Inicia la IA
