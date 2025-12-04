# Configuração do Kaggle

Para usar esta aplicação, você precisa configurar suas credenciais do Kaggle.

## Opção 1: Arquivo de Configuração (Recomendado)

1. Acesse https://www.kaggle.com/ e faça login
2. Vá em **Account** → **API** → **Create New Token**
3. Isso baixará um arquivo `kaggle.json`
4. Coloque o arquivo em `~/.kaggle/kaggle.json`:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

O arquivo `kaggle.json` deve ter o formato:
```json
{
  "username": "seu_usuario_kaggle",
  "key": "sua_chave_api_kaggle"
}
```

## Opção 2: Variáveis de Ambiente

Você também pode definir as variáveis de ambiente:

```bash
export KAGGLE_USERNAME=seu_usuario_kaggle
export KAGGLE_KEY=sua_chave_api_kaggle
```

## Verificação

Para verificar se está configurado corretamente, execute:

```python
import kagglehub
path = kagglehub.dataset_download("mkashifn/nbaiot-dataset")
print("Dataset baixado em:", path)
```

Se funcionar sem erros, está tudo configurado! ✅





