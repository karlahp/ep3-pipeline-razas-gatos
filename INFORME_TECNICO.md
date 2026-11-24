# Informe técnico — EP3: Pipeline integrado de clasificación de razas de gatos

## Problema y objetivo

Se plantea una herramienta de apoyo para el registro veterinario: dada una fotografía, el sistema propone una raza y el personal confirma o corrige la sugerencia. Se comparan dos arquitecturas de clasificación multiclase, una red MLP y una CNN. La recomendación no es para diagnóstico ni decisión automática.

## Datos y preparación

Se utilizó Oxford-IIIT Pet y una variante de cinco razas de gatos visualmente similares (Birman, Ragdoll, Siamese, Persian y Maine Coon), con 184 imágenes por raza: 920 imágenes en total. La partición fue estratificada (70% entrenamiento, 15% validación, 15% prueba: 644, 138 y 138 imágenes). Las imágenes se redimensionan a 64×64 RGB. La normalización se ajusta únicamente con las imágenes de entrenamiento de cada fold. El aumento aleatorio se aplica solo durante entrenamiento y se comparó como factor de configuración.

El EDA incluye balance de clases, ejemplos, dimensiones y proporciones, modos de color e imágenes corruptas, además de medidas visuales por raza. Las figuras y análisis detallados están en el notebook.

## Modelos y evaluación

La MLP aplana la imagen de entrada y utiliza una capa densa oculta; la CNN usa tres bloques convolucionales con max pooling y una capa de promedio global. Se emplearon validación cruzada estratificada de cinco folds en desarrollo, búsqueda de hiperparámetros de un factor a la vez y Early Stopping. El conjunto de prueba permaneció separado de la selección y se utilizó en la evaluación final con cinco semillas.

### Resultados finales en test

| Métrica (media ± desviación estándar entre semillas) | MLP | CNN |
|---|---:|---:|
| F1 macro | 0,376 ± 0,013 | 0,376 ± 0,053 |
| Accuracy | 0,378 | 0,386 |
| ROC-AUC macro | 0,704 | 0,742 |
| Recall mínimo medio | 0,259 | 0,206 |

El KPI principal (F1 macro ≥ 0,40) no se alcanzó: ambas arquitecturas obtuvieron 0,376 de media. El recall mínimo medio sí superó 0,20 en ambas, aunque su estabilidad por semilla difiere. Los resultados no justifican uso automático.

### Regularización adicional

El experimento controlado de L2 y Batch Normalization se realizó con cinco folds, una semilla y un presupuesto común de 60 épocas como máximo y paciencia 10. Los F1 macro medios fueron: MLP base 0,390; MLP+L2 0,403; MLP+Batch Normalization 0,413; CNN base 0,386; CNN+L2 0,395; CNN+Batch Normalization 0,294. Son resultados exploratorios; los folds comparten datos de entrenamiento, se usó una semilla y no se guardaron curvas de aprendizaje para estas seis configuraciones. El notebook presenta además la desviación estándar y otras métricas.

## Auditoría, XAI y límites

El notebook incluye matrices de confusión, análisis por raza, evaluación de abstención, mapas Grad-CAM para la CNN, saliencia para la MLP y una comparación de la atención con las máscaras del dataset. Estas técnicas ayudan a inspeccionar patrones, pero no prueban causalidad. El alcance se limita a cinco razas, hay pocas imágenes por clase y la selección de hiperparámetros fue acotada. La comparación final de modelos optimizados no iguala todos los límites de entrenamiento; se presenta por separado un experimento exploratorio controlado con presupuesto común.

La recomendación de la CNN es provisional: su AUC fue mayor que el de la MLP en las cinco semillas, pero ambas empataron en F1 y accuracy similar. La MLP sigue siendo una alternativa válida. Se propone usar cualquier modelo solo como apoyo, con abstención y confirmación humana, y estudiar más datos, transfer learning, calibración y validación con un conjunto externo.

## Reproducibilidad

El notebook conserva las salidas de la corrida, semillas y protocolo, y descarga Oxford-IIIT Pet desde su fuente oficial. La corrida registrada se ejecutó en CPU con Python 3.13.18 y TensorFlow 2.21.0. Repetir el entrenamiento completo puede tomar varias horas en CPU. No se distribuyen los datos ni los modelos entrenados; se descargan o generan al ejecutar el notebook.

## Referencias

- Parkhi, O. M., Vedaldi, A., Zisserman, A. y Jawahar, C. V. (2012). *Cats and Dogs*. CVPR. [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV.
- Srivastava, N. et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. Journal of Machine Learning Research, 15, 1929–1958.
- Kingma, D. P. y Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. ICLR.
- Chapman, P. et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*.
