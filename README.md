
**Descripcion:**

﻿Este es un script de **Python 3** que implementa metodos de interaccion con hasta 14 inteligencias artificiales (IA) utilizando la estructura de Ollama.  
La IA puede responder a preguntas y mantener conversaciones con el usuario.  
IA disponibles:  
`Llama 3 8B` 4.7GB, `Llama 3 70B`  40GB,  
`Phi 3 Mini 3.8B` 2.3GB, `Phi 3 Medium 14B` 7.9GB,  
`Gemma 2B` 1.4GB, `Gemma 7B` 4.8GB,  
`Mistral 7B` 4.1GB,  
`Moondream 2 1.4B` 829MB,  
`Neural Chat 7B` 4.1GB,  
`Starling 7B`  4.1GB,  
`Code Llama 7B` 3.8GB,  
`Llama 2 Uncensored 7B` 3.8GB,  
`LLaVA  7B` 4.5GB,  
`Solar 10.7B` 6.1GB

**Funcionalidades**  
  El script incorpora detección y activación de GPU para acelerar el procesamiento de información. Si no es posible su activación, notificará que se utilizará únicamente la CPU de la computadora.  
Esta capacidad puede ser identificada a lo largo del historial de chat mediante el color preasignado de cada respuesta de la IA: ROJO para uso de CPU y CYAN para uso con GPU.  
Si por alguna razón no ha podido ser activada la GPU, presentará un detalle del problema y brindando la solución que tendrá que ver con la instalación de CUDA, siendo fundamental para el correcto desarrollo de detección y funcionalidad del código.  
  También incluirá una fecha al inicio y un registro de la hora en cada respuesta de la inteligencia artificial (IA).  


El script utiliza las siguientes librerías:

* Ollama y Langchain_Community.LLMS, necesarias para el correcto funcionamiento de la inteligencia artificial (IA).
* Sys y Os, utilizados para funciones básicas como limpiar pantalla.
* Torch, utilizado para administrar la GPU y garantizar un rendimiento óptimo.
* Datetime, necesario para incorporar fecha y hora en las respuestas de la IA.
* Colorama, utilizado para brindar notificaciones destacadas y contraste en el output.
 
En la línea de comandos:
```
#-------------------------
MOTOR_IA = "Llama3"  # Modifica si quieres usar otra IA
-------------------------
```
Puedes asignar la IA con la que deseas trabajar, asignando el nombre de la misma desde las notas en el mismo código. Debes tener previamente instalada la IA preinstalada en Ollama, lo cual se explicará en la sección REQUERIMIENTOS.  
La aplicación detectará automáticamente la IA elegida y pré-asignara el nombre a tu IA.  
Nota: Estas modificaciones no afectan internamente a la IA, sino que son mensajes pre-programados que se presentarán al usuario como si fuera la IA, para poder tener un entorno más amigable en el contexto.  
A continuación, la aplicación se desarrolla ejecutando las funciones según se asignó por el creador, lo que da paso a la interacción con la IA de forma tradicional.  
Nota: Este script es una implementación básica de inteligencia artificial conversacional utilizando la arquitectura de Ollama y algunas funciones adicionales para personalizar la salida en pantalla.  

**Requerimientos**
* Activar GPU, Requiere tener instalado CUDA https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
* Instalar ollama desde https://ollama.com de lo contrario este script python no funcinara.
* Ejecutar `OLLAMA` y luego Ejecutar `terminal`: escribir "ollama run llama3"  esto instalara llama3 y lo correra, o el codigo  a continuacion para la IA que queiras
  ollama run llama3
  ollama run llama3:70b
  ollama run phi3
  ollama run phi3:medium
  ollama run gemma:2b
  ollama run gemma:7b
  ollama run mistral
  ollama run moondream
  ollama run neural-chat
  ollama run starling-lm    
  ollama run codellama  
  ollama run llama2-uncensored
  ollama run llava
  ollama run solar
  ```
  Al correr este codigo automaticamente detectara si la IA esta instalada o no, y la instalara.
  ```
  Nota: El modelo IA se guarda en C:\Users\XXXX\models
  
* Instalar librerias:  
  python.exe -m pip install upgrade pip
  pip install ollama
  pip install langchain_community.llms
  pip instlla torch
  pip install colorama
```
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
```
* Modificar la variable `MOTOR_IA`por la que has descargado.  
   Puedes usar: llama3, llama3:70b, phi3, phi3:medium, gemma:2b, gemma:7b, mistral, moondream, neural-chat, starling-lm, codellama, llama2-uncensored, llava, solar

**Uso**
Una ves descargado Ollama e instalado la IA a tu eleccion, y haber instalado Cuda y las librerias del script.  
solo queda modificar la linea de comando:  
```
# Puedes usar: llama3, llama3:70b, phi3, phi3:medium, gemma:2b, gemma:7b, mistral, moondream, neural-chat, starling-lm, codellama, llama2-uncensored, llava, solar  

#--------------------------
MOTOR_IA = "llama3" # MODIFICALO SI QUIERES USAR OTRA IA
#--------------------------
```
Luego solo queda correr el script, utilizando cualquiera de los siguientes codigos:
`python.exe chatbot.py`
`python chatbot.py`
`py chatbot.py`

# Codigo por Pablo Otero. 18/06/2024 Python 3.11.9 

# Ejemplo:  
Foto1:  ![2024-06-18_04-07](https://github.com/pablootero-cl/AI-ChatBot---Ptrhon3/assets/172928670/4c51f93f-ea55-41f4-bfcd-f201565b99ae)

Foto2:  ![2024-06-18_04-11](https://github.com/pablootero-cl/AI-ChatBot---Ptrhon3/assets/172928670/caab8d02-7d7f-4f2f-90ab-84567f7a0ed9)

