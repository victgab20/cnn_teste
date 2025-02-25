import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

path = r"C:\Users\victo\Downloads\corpus\labeled\\"  

# Lista de arquivos a processar
datasets = ['train_apps.pkl', 'test_apps.pkl', 'dev_apps.pkl']

for dataset in datasets:
    # Carregar o arquivo
    df = pd.read_pickle(path + dataset)
    print(f'Processando {dataset}...')
    print(df.head())

    # Aplicar undersampling
    rus = RandomUnderSampler(random_state=2021)
    df = df.dropna()
    text = df[['text']]
    label = df[['helpfulness']]
    features = []

    text, label = rus.fit_resample(text, label)

    for t, l in zip(text.values, label.values):  
        features.append({'text': t[0], 'helpfulness': l[0]})

    # Criar DataFrame balanceado
    bdf = pd.DataFrame(features, columns=['text', 'helpfulness'])

    # Nome do arquivo de saída
    output_file = path + 'balanced_' + dataset.replace('.pkl', '.csv')

    # Salvar no CSV
    bdf.to_csv(output_file, index=False)

    # Verificar o CSV salvo
    print(f'Arquivo salvo: {output_file}')
    print(pd.read_csv(output_file).head())

print("Processamento concluído!")
