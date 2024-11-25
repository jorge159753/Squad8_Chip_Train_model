from ultralytics import YOLO
import os

# Função para treinar o modelo
def treinar_modelo(dataset_yaml, epochs=5, batch_size=8, img_size=416, workers=16):

    # Carregar o modelo pré-treinado
    model = YOLO() #Adicionar qualquer arquivo .pt como pré-treinamento

    # Treinamento
    model.train(
        data=dataset_yaml,         # Caminho do dataset (YAML)
        epochs=epochs,            
        batch=batch_size,          
        imgsz=img_size,
        workers=workers,            
        project="volumeYOLO",      # Nome do projeto
        name="modelo_treinado",              
        save=True                 
    )

    print("Treinamento concluído.")


# Função principal para iniciar o treinamento
def main():
    # Caminho para o arquivo de configuração do dataset
    dataset_yaml = "dataset.yaml"

    # Iniciar o treinamento do modelo
    treinar_modelo(dataset_yaml, epochs=5, batch_size=8, img_size=416, workers=16)


# Executar a função principal
if __name__ == "__main__":
    main()
