from ultralytics import YOLO
import os
import shutil

def preparar_dados(dataset_dir, output_dir):
    
    # Cria estrutura de treino e validação
    train_dir = os.path.join(output_dir, "images/Train")
    val_dir = os.path.join(output_dir, "images/val")
    labels_train_dir = os.path.join(output_dir, "labels/Train")
    labels_val_dir = os.path.join(output_dir, "labels/val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)

    # Separar imagens em treino e validação (80% treino, 20% validação)
    images = [f for f in os.listdir(dataset_dir) if f.endswith(('.jpg', '.png'))]
    annotations = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in images]
    train_size = int(0.8 * len(images))

    for i, (img, label) in enumerate(zip(images, annotations)):
        if i < train_size:
            shutil.copy(os.path.join(dataset_dir, img), train_dir)
            shutil.copy(os.path.join(dataset_dir, label), labels_train_dir)
        else:
            shutil.copy(os.path.join(dataset_dir, img), val_dir)
            shutil.copy(os.path.join(dataset_dir, label), labels_val_dir)

    # Criação do arquivo YAML
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write(f"train: {os.path.abspath(train_dir)}\n")
        f.write(f"val: {os.path.abspath(val_dir)}\n")
        f.write("nc: 2\n")
        f.write("names: ['car', 'accident']\n")
    
    return yaml_path

def excluir_modelo_antigo(output_dir):

    #Exclui o modelo treinado anterior, se existir, para permitir a substituição pelo novo modelo.

    modelo_antigo_dir = os.path.join(output_dir, "modelo_melhorado", "weights", "best.pt")
    if os.path.exists(modelo_antigo_dir):
        print(f"Excluindo modelo antigo em: {modelo_antigo_dir}")
        os.remove(modelo_antigo_dir)

def treinar_modelo(data_yaml, output_dir, epochs=10, batch_size=16, img_size=640):
    
    # Inicializar o modelo best pré-treinado
    model = YOLO("best01.pt")
    
    # Excluir modelo anterior, se existir
    excluir_modelo_antigo(output_dir)
    
    # Treinamento
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=output_dir,
        name="best_update",
        exist_ok=True
    )
    
    # Retornar o caminho para o modelo treinado
    best_model_path = os.path.join(output_dir, "best_update", "weights", "best.pt")
    print(f"Treinamento concluído! Modelo salvo em {best_model_path}")
    return best_model_path

# Caminhos
images_Train = os.path.join(VOLUME_FRAME_PATH)
val_Train = os.path.join(VOLUME_FRAME_TREINAMENTO)
dataset_dir = os.path.join("dataset") 
output_dir = os.path.join("volumeYOLO")  #Dirotório de saida dos dados

# Execução
if __name__ == "__main__":
    data_yaml = preparar_dados(dataset_dir, output_dir)
    best_model = treinar_modelo(data_yaml, output_dir)
    if best_model:
        print(f"Modelo final salvo em: {best_model}")
