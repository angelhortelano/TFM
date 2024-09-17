# 📖 Manual de Usuario - Chat Legal basado en RAG

## 🛠️ Descripción del Proyecto
Este proyecto es una aplicación **RAG (Retrieval-Augmented Generation)** que permite consultar información relacionada con el **Código de Tráfico y Seguridad Vial de España**. Utiliza técnicas avanzadas de procesamiento del lenguaje natural (NLP) 🧠 para recuperar información relevante y generar respuestas precisas basadas en artículos y leyes específicas del BOE 📜. El sistema está optimizado para ofrecer respuestas claras y útiles, basadas en el contexto de la consulta del usuario 💬.

## 🖥️ ¿Cómo funciona la interfaz?

### 🖱️ Interfaz de Usuario
La aplicación presenta una interfaz interactiva en **Streamlit** 🖼️, diseñada para que los usuarios puedan hacer preguntas relacionadas con el **Código de Tráfico y Seguridad Vial** 🚦. El flujo básico es el siguiente:

1. **Entrada de Preguntas**: En la parte inferior de la pantalla 📝, verás un campo de texto donde puedes introducir una pregunta relacionada con el **Código de Tráfico**.
2. **Visualización del Historial de Conversación**: En la parte principal de la interfaz, se muestran tanto las preguntas que has hecho como las respuestas generadas por el sistema 🔄.
3. **Respuestas Contextualizadas**: Las respuestas se generan utilizando fragmentos relevantes del **BOE** 📚 y están redactadas de manera clara para que sean fácilmente comprensibles 🧐. El sistema también indica los artículos o leyes relacionadas, mencionando el BOE correspondiente 📖.

### ⏳ Tiempos de Carga
La **primera vez que arrancas la aplicación**, es posible que tarde entre **1 a 2 minutos** ⌛ en estar completamente lista. Esto se debe a la carga de los modelos y la recuperación de los embeddings almacenados 🗂️.

## 🔗 Link de la Aplicación
El enlace a la aplicación es el siguiente:  
[**Enlace a la App**](https://app-trafico-869260115209.europe-southwest1.run.app/) 🌐

Haz clic en el enlace para acceder a la aplicación y empezar a interactuar con el modelo 🤖.

## 🤖 Ejemplos de Preguntas

Aquí te mostramos algunos ejemplos de preguntas que puedes hacer al sistema:

1. **Consultas sobre límites de velocidad**:
   - *"¿Cuáles son los límites de velocidad en vías urbanas?"* 🛣️
   - *"¿Qué dice el Artículo 48 sobre las velocidades máximas fuera de poblado?"* 📜

2. **Dudas sobre como actuar**:
   - *"¿Con que edad puedo sacarme el permiso B de conducir?"* 🚗➡️🚗
   - *"¿Qué dice el código sobre las señales de stop?"* 🛑

3. **Consultas generales sobre el Código de Tráfico**:
   - *"¿Cuál es la ley que regula el uso del cinturón de seguridad?"* 🔒
   - *"¿Puedo circular con una bicicleta por una autopista?"* 🏍️

## ℹ️ Consideraciones
- **Primera vez de arranque**: Como se mencionó antes, la primera vez que se inicie la aplicación puede tardar entre 1 a 2 minutos ⏲️ debido a la carga de los modelos y datos.
- **Respuestas legales**: Aunque el sistema proporciona respuestas basadas en el **Código de Tráfico y Seguridad Vial** 📚, es recomendable consultar con un experto 👨‍⚖️ antes de tomar decisiones legales basadas en las respuestas generadas.

## ❓ Preguntas Frecuentes
- **¿Qué hago si el sistema no encuentra una respuesta?**  
   Si el sistema no puede encontrar una respuesta en el contexto proporcionado, intentará guiarte para que reformules la pregunta 🔄 o indicará que no dispone de la información en el momento.
