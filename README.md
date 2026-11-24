# EP3 — Pipeline integrado para clasificar razas de gatos

Proyecto académico de aprendizaje profundo que integra preparación de datos, entrenamiento, optimización, evaluación, auditoría y explicabilidad para clasificar imágenes de cinco razas de gatos visualmente similares.

> **Resultado principal:** la MLP y la CNN obtuvieron F1 macro de prueba 0,376. Ninguna alcanzó el KPI principal (F1 macro ≥ 0,40). La CNN tuvo mayor ROC-AUC en las cinco semillas, pero su recomendación es provisional. El sistema se plantea solo como apoyo con confirmación humana.

## Índice

- [Problema y objetivo](#problema-y-objetivo)
- [Trazabilidad entre entregas](#trazabilidad-entre-entregas)
- [Datos y alcance](#datos-y-alcance)
- [Pipeline y preparación](#pipeline-y-preparación)
- [Modelos y protocolo experimental](#modelos-y-protocolo-experimental)
- [Resultados](#resultados)
- [Auditoría y explicabilidad](#auditoría-y-explicabilidad)
- [Conclusiones y límites](#conclusiones-y-límites)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Cómo reproducir](#cómo-reproducir)
- [Referencias](#referencias)

## Problema y objetivo

En una clínica veterinaria, el ingreso de un paciente incluye registrar sus datos y antecedentes. La raza puede orientar la organización de la ficha y la consulta de información preventiva, pero identificarla visualmente no siempre es sencillo: algunas razas comparten rasgos y las fotografías cambian en iluminación, postura, encuadre y fondo. El problema abordado es si un clasificador puede proponer una de cinco razas a partir de una fotografía y ayudar a completar ese registro.

La solución se plantea como **apoyo al personal veterinario**: el modelo presenta una sugerencia y la persona la confirma o corrige. No reemplaza el criterio profesional ni realiza diagnóstico o decisión clínica automática. Además, el proyecto no midió si la propuesta reduce el tiempo de registro o los errores en una clínica real; ese beneficio es una hipótesis de uso que requeriría validación con usuarios y datos reales.

El objetivo técnico fue construir un pipeline modular y reproducible para comparar una red neuronal densa (MLP) y una red convolucional (CNN), seleccionar configuraciones con datos de validación, estudiar regularización y analizar errores, sesgos y explicaciones. La tarea experimental se limita a cinco razas de gatos visualmente similares; por ello, la salida del modelo no cubre gatos mestizos ni razas fuera de esas cinco clases.

### Indicadores objetivo

| KPI | Meta | Resultado en prueba | Estado |
|---|---:|---:|---|
| F1 macro | ≥ 0,40 | MLP 0,376; CNN 0,376 | No alcanzado |
| Recall mínimo medio entre razas | ≥ 0,20 | MLP 0,259; CNN 0,206 | Alcanzado en promedio |

Los KPI se mantienen desde la EP1 según la trazabilidad documentada en la EP2. El umbral de F1 es una referencia técnica y no un umbral operativo validado con una clínica. El notebook informa también la variación por semilla y el detalle por raza.

## Trazabilidad entre entregas

- **EP1, versión original:** clasificación binaria de perros y gatos.
- **EP1, versión rehecha:** se trabajó con cinco razas de gatos parecidas y balanceadas; esta variante y su partición son la base para comparar entre entregas.
- **EP2:** se incorporó una CNN al problema de clasificación por raza y se mantuvo la comparación con la MLP.
- **EP3:** se integró el proceso en un pipeline con validación cruzada, búsqueda controlada, regularización, evaluación final, auditoría de errores y XAI.

La introducción del notebook contiene el detalle de la trazabilidad y aclara qué versión de EP1 se usa como referencia.

## Datos y alcance

Se utilizó el [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), que contiene 7.349 imágenes anotadas de 37 razas de perros y gatos. Para este trabajo se seleccionaron cinco razas de gatos visualmente similares:

- Birman
- Ragdoll
- Siamese
- Persian
- Maine Coon

Se usaron 184 imágenes por raza (920 en total). Los resultados de este repositorio corresponden a **estas cinco razas**, no a las 37 del conjunto ni al subconjunto de razas visualmente distintas.

## Metodología utilizada: CRISP-DM

El proyecto organiza sus etapas siguiendo CRISP-DM. La aplicación es completa hasta evaluación; el despliegue se presenta como propuesta porque no se probó en una clínica.

| Fase CRISP-DM | Aplicación en este proyecto |
|---|---|
| Comprensión del negocio | Contexto de registro veterinario, objetivo de apoyo, alcance y KPI técnicos. |
| Comprensión de los datos | Carga de Oxford-IIIT Pet, EDA de balance, ejemplos, dimensiones, calidad y patrones visuales. |
| Preparación de los datos | Selección balanceada, división estratificada, redimensionamiento, normalización sin fuga y aumento solo en entrenamiento. |
| Modelado | Construcción y entrenamiento de MLP y CNN con búsqueda y regularización. |
| Evaluación | Validación cruzada, selección, prueba final, métricas, auditoría y XAI. |
| Despliegue | Propuesta de uso con abstención y confirmación humana; no se desplegó en producción. |

El detalle de cada fase y sus secciones correspondientes está en la sección 1.6 del notebook.
## Pipeline y preparación

El pipeline cubre configuración, descarga y lectura de datos, muestreo, partición, preprocesamiento, EDA, construcción de modelos, validación cruzada, selección, evaluación final, auditoría y explicabilidad.

La división es estratificada: 70% entrenamiento, 15% validación y 15% prueba (644, 138 y 138 imágenes). En la búsqueda se usan cinco folds estratificados sobre el conjunto de desarrollo. Las imágenes se redimensionan a 64×64 píxeles RGB. La normalización se adapta solo con los datos de entrenamiento de cada fold para reducir el riesgo de fuga de información. El aumento aleatorio se aplica durante entrenamiento y se evalúa como factor de configuración.

El EDA revisa balance por raza, ejemplos, dimensiones y proporciones de las imágenes, modos de color, archivos corruptos y características visuales. El notebook presenta las figuras y el análisis detallado.

## Modelos y protocolo experimental

- **MLP:** imagen aplanada (12.288 valores) → Dense(128, ReLU) → Dropout → salida softmax de cinco clases.
- **CNN:** tres bloques Conv2D + MaxPooling (32, 64 y 128 filtros) → GlobalAveragePooling → Dense(64, ReLU) → Dropout → salida softmax.

La búsqueda cambia un factor a la vez y utiliza los mismos folds y semilla para las configuraciones comparadas. Se seleccionaron MLP_lr1e-3 y CNN_sin_aumento por su F1 medio en validación. Luego se evaluaron en el test separado con cinco semillas.

La comparación final optimizada no iguala todos los límites de entrenamiento de la MLP y la CNN. El notebook declara esa limitación y presenta además un experimento exploratorio con presupuesto común (60 épocas como máximo y paciencia 10) para modelos base y variantes L2/Batch Normalization.

## Resultados

### Selección por validación cruzada

| Configuración elegida | F1 macro medio ± DE | Accuracy media |
|---|---:|---:|
| MLP (MLP_lr1e-3) | 0,407 ± 0,039 | 0,412 |
| CNN (CNN_sin_aumento) | 0,475 ± 0,058 | 0,480 |

La selección siguió la regla predefinida de mayor F1 medio. La ventaja de la CNN en validación no se mantuvo en el test, por lo que no se interpreta como evidencia de superioridad final.

### Evaluación final en test

| Métrica (media ± desviación estándar entre cinco semillas) | MLP | CNN |
|---|---:|---:|
| F1 macro | 0,376 ± 0,013 | 0,376 ± 0,053 |
| Accuracy | 0,378 | 0,386 |
| ROC-AUC macro | 0,704 | 0,742 |
| Recall mínimo medio | 0,259 | 0,206 |

El F1 macro promedio es igual al redondear. La CNN tuvo mayor ROC-AUC en las cinco semillas, pero también mayor variabilidad de F1. El KPI principal no se alcanzó; por ello no se propone automatizar el registro.

### Experimento controlado de regularización

Se probaron L2 y Batch Normalization por separado, con cinco folds y una semilla:

| Arquitectura | Base | + L2 (1e-4) | + Batch Normalization |
|---|---:|---:|---:|
| MLP, F1 macro medio ± DE | 0,390 ± 0,029 | 0,403 ± 0,029 | 0,413 ± 0,047 |
| CNN, F1 macro medio ± DE | 0,386 ± 0,079 | 0,395 ± 0,085 | 0,294 ± 0,156 |

Son resultados exploratorios: los folds comparten imágenes de entrenamiento y se usó una sola semilla. No se probaron combinaciones de L2 y Batch Normalization ni otros valores de L2, y no se guardaron curvas de aprendizaje para estas seis configuraciones. Estos resultados no cambiaron la selección final ni consultaron el test.

## Auditoría y explicabilidad

El notebook incluye matrices de confusión, métricas por raza, análisis de errores, abstención por confianza, Grad-CAM para la CNN y mapas de saliencia para la MLP. También compara la atención con las máscaras de animal del dataset y prueba cambios de fondo. Estas herramientas ayudan a inspeccionar el comportamiento, pero no demuestran causalidad ni garantizan que el modelo use rasgos apropiados en cada caso.

## Conclusiones y límites

El pipeline permitió comparar dos arquitecturas, estudiar regularización y conservar evidencia de los resultados. La CNN es una recomendación **provisional** por su ROC-AUC más alto en las cinco semillas; no supera a la MLP en F1 y su accuracy es similar. La MLP sigue siendo una alternativa válida y más estable en F1 entre semillas.

El alcance se limita a cinco razas con 184 imágenes cada una. El KPI de F1 no se cumplió y la comparación final optimizada no controla todos los factores de entrenamiento. La búsqueda fue acotada; L2 y Batch Normalization se probaron con una semilla y sin curvas. No se recomienda uso automático. Los siguientes pasos son ampliar los datos, probar transfer learning, calibrar probabilidades y validar con un conjunto externo y una clase para razas no incluidas.

## Estructura del repositorio

    ep3-pipeline-razas-gatos/
    ├── .gitignore
    ├── README.md
    ├── INFORME_TECNICO.md
    ├── data/
    ├── images/
    ├── models/
    ├── resultados/
    └── notebooks/
        └── EP3_Pipeline_Razas_Gatos_GitHub.ipynb

Las carpetas data/, images/, models/ y resultados/ incluyen una nota README.md que explica su uso. No contienen el dataset ni pesos/artefactos de entrenamiento: los directorios sirven para ordenar archivos locales generados al reproducir el notebook y el .gitignore evita subir esos archivos. Las figuras de la corrida ya están incrustadas en las salidas del notebook.

## Cómo reproducir

1. Clona o descarga este repositorio.
2. Abre notebooks/EP3_Pipeline_Razas_Gatos_GitHub.ipynb en Google Colab o Jupyter.
3. Asegura acceso a Internet para que el notebook descargue Oxford-IIIT Pet e instala/disponibiliza las dependencias importadas en las primeras celdas.
4. Ejecuta las celdas desde el inicio y en orden. El notebook incluye checkpoints locales opcionales para reanudar algunas etapas si se interrumpe la sesión.

La corrida guardada se ejecutó con Python 3.13.18 y TensorFlow 2.21.0 en CPU, sin GPU. Reproducir el entrenamiento completo puede tardar varias horas en CPU. Abrir el notebook en GitHub permite revisar código, métricas y figuras guardadas sin volver a entrenar.

## Referencias

- Parkhi, O. M., Vedaldi, A., Zisserman, A. y Jawahar, C. V. (2012). *Cats and Dogs*. CVPR. [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV.
- Srivastava, N. et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. *Journal of Machine Learning Research*, 15, 1929–1958.
- Kingma, D. P. y Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR.
- Chapman, P. et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*.
- La bibliografía completa y los enlaces a documentación de TensorFlow/Keras y scikit-learn están en la sección 17 del notebook.

