import os
import shutil
from ultralytics import YOLO

# Diretório onde os modelos serão armazenados
BASE_PATH = os.path.join('Squad8_Chip_Train_model/app')
VOLUME_YOLO = os.path.join(BASE_PATH, 'volumeYolo')

# Certifica-se de que o diretório VOLUME_YOLO existe
if not os.path.exists(VOLUME_YOLO):
    os.makedirs(VOLUME_YOLO)

def limpar_diretorio(diretorio, pasta_preservada="weights"):
    # Verifica se o diretório existe antes de tentar limpar
    if not os.path.exists(diretorio):
        print(f"Diretório {diretorio} não encontrado!")
        return
    
    # Exclui tudo, mas mantém a pasta especificada (por padrão, 'weights')
    for arquivo in os.listdir(diretorio):
        caminho_arquivo = os.path.join(diretorio, arquivo)

        # Se for a pasta 'weights', não exclua nem seus arquivos
        if arquivo == pasta_preservada and os.path.isdir(caminho_arquivo):
            print(f"Preservando a pasta '{pasta_preservada}'.")
            continue  # Pula a pasta 'weights', não faz nada com ela

        # Caso contrário, remove o arquivo ou a pasta
        if os.path.isdir(caminho_arquivo):
            print(f"Removendo subpasta: {caminho_arquivo}")
            shutil.rmtree(caminho_arquivo)  # Remove subpastas
        else:
            print(f"Removendo arquivo: {caminho_arquivo}")
            os.remove(caminho_arquivo)  # Remove outros arquivos

# Função para obter o modelo mais recente
def get_latest_best():
    # Filtra apenas arquivos com extensão '.pt' (ou outra extensão de modelos)
    model_files = [os.path.join(VOLUME_YOLO, f) for f in os.listdir(VOLUME_YOLO) if f.endswith('.pt')]
    
    if not model_files:
        raise FileNotFoundError("Nenhum modelo encontrado no diretório.")

    # Retorna o modelo mais recente
    return max(model_files, key=os.path.getctime)


# Função para inicializar o modelo (carregar um modelo externo ou treinar do zero)
def initialize_model(pretrained_model_path=None):
    # Se um caminho para um modelo externo for fornecido, tente carregá-lo
    if pretrained_model_path:
        print(f"Carregando o modelo externo pré-treinado: {pretrained_model_path}")
        if os.path.exists(pretrained_model_path):
            return YOLO(pretrained_model_path)  # Carrega o modelo a partir do caminho fornecido
        else:
            print(f"Erro: O modelo externo em {pretrained_model_path} não foi encontrado.")
            raise FileNotFoundError(f"Modelo externo não encontrado em: {pretrained_model_path}")
    
    # Caso contrário, inicializa um novo modelo (se não passar modelo externo)
    print("Nenhum modelo externo fornecido, treinando a partir do zero...")
    return YOLO()


# Função para treinar o modelo
def treinar_modelo(dataset_yaml, epochs=5, batch_size=8, img_size=416, workers=16, pretrained_model_path=None):
    # Inicializar o modelo ou carregar o modelo externo (pré-treinado)
    model = initialize_model(pretrained_model_path=pretrained_model_path)

    # Treinamento
    model.train(
        data=dataset_yaml,         # Caminho do dataset (YAML)
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        workers=workers,
        project=VOLUME_YOLO,        # Nome do projeto
        name="modelo_treinado",     # Subdiretório onde o modelo será salvo
        save=True,                  # Garante que o modelo será salvo
        save_period=0,              # Não salva checkpoints intermediários
        save_frames=False,          # Desabilita o salvamento das imagens dos batches
        save_json=False,            # Desabilita o salvamento de dados de treinamento em JSON
        exist_ok=True               # Se já existir, não cria outra pasta
    )

    # Verificar arquivos após o treinamento
    print("Arquivos após o treinamento:")
    for root, dirs, files in os.walk(os.path.join(VOLUME_YOLO, "modelo_treinado")):
        for file in files:
            print(f"Arquivo encontrado: {file}")

    # Verificar se o best.pt foi criado
    best_model_path = os.path.join(VOLUME_YOLO, "modelo_treinado", "best.pt")
    if os.path.exists(best_model_path):
        print("Modelo 'best.pt' encontrado, mantendo-o.")
    else:
        print("Modelo 'best.pt' não encontrado!")

    # Limpar o diretório e manter apenas o arquivo 'best.pt'
    limpar_diretorio(os.path.join(VOLUME_YOLO, "modelo_treinado"))

    print("Treinamento concluído.")


import os

def main():
    # Caminho para o arquivo de configuração do dataset
    dataset_yaml = "dataset.yaml"
    
    # Caminho do modelo externo pré-treinado (modelo vindo de fora)
    pretrained_model_path = ""  # Exemplo de caminho
    
    # Verificação se o modelo pré-treinado existe
    if not os.path.exists(pretrained_model_path):
        print(f"Erro: O modelo pré-treinado '{pretrained_model_path}' não foi encontrado.")
        return  # Sai da função main se o modelo não for encontrado
    
    # Iniciar o treinamento do modelo com o modelo pré-existente
    treinar_modelo(dataset_yaml, epochs=5, batch_size=8, img_size=416, workers=16, pretrained_model_path=pretrained_model_path)

# Executar a função principal
if __name__ == "__main__":
    main()
