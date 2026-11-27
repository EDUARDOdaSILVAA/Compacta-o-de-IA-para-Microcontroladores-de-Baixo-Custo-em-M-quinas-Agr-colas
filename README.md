# TinyML — Compactação de Redes Neurais para Microcontroladores

## Sobre o projeto

Este projeto investiga a possibilidade de reduzir o tamanho de modelos de inteligência artificial sem comprometer significativamente a qualidade das suas classificações, visando sua futura execução em microcontroladores de baixo custo utilizados em aplicações agrícolas.

## Experimento preliminar

Foi desenvolvido um experimento em Python utilizando PyTorch para comparar duas redes neurais:

- **Rede original:** 16 filtros na primeira camada convolucional e 32 na segunda.
- **Rede reduzida:** 8 filtros na primeira camada e 16 na segunda.

A redução foi feita por poda estruturada, preservando filtros selecionados com base na magnitude de seus pesos.

Foram utilizados sinais artificiais de quatro classes: normal, pista interna, esfera e pista externa. Esses sinais não representam medições reais de falhas.

## Como executar no Linux

Com Python 3 instalado, na pasta do projeto, execute:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python teste_preliminar.py
```

O script gera os arquivos `pesos_referencia.pt`, `pesos_podado.pt` e `resultados_teste.json`.

## Resultados preliminares

| Métrica | Original | Reduzida |
|---|---:|---:|
| Parâmetros | 2.852 | 788 |
| Arquivo de pesos | 14.529 bytes | 6.161 bytes |
| Macro-F1 | 1,00 | 1,00 |

A redução foi de 72,37% na quantidade de parâmetros e 57,60% no tamanho do arquivo de pesos, sem perda observada na qualidade da classificação dos dados artificiais testados.

Os resultados não comprovam desempenho equivalente em dados reais ou execução no Raspberry Pi Pico.

## Próximas etapas

- Testar sinais reais da base CWRU.
- Avaliar diferentes níveis de compactação.
- Investigar quantização INT8.
- Verificar a execução no Raspberry Pi Pico.


# Executando o protótipo TinyML no Windows

Este guia mostra como executar no **Windows 10 ou 11**, utilizando **PowerShell**, o experimento preliminar de compactação de redes neurais do projeto TinyML.

> **Escopo do experimento:** o script `teste_preliminar.py` utiliza sinais **sintéticos** para comparar uma CNN original com uma CNN menor. Ele **não utiliza os dados reais da CWRU**, **não executa no Raspberry Pi Pico** e **não exporta modelos `.tflite`**.

## 1. Pré-requisitos

- Python 3 instalado (recomendado **Python 3.11 ou 3.12**), com o inicializador `py` disponível no terminal;
- PowerShell (já incluído no Windows);
- Internet para instalar os pacotes na primeira execução;
- Arquivos `teste_preliminar.py` e `requirements.txt` na mesma pasta.

É possível baixar o Python em <https://www.python.org/downloads/windows/>. Não é necessária GPU dedicada: o script funciona na CPU.

Verifique a instalação com:

```powershell
py --version
```

Se `py` não for reconhecido, tente `python --version` e substitua `py` por `python` no comando de criação do ambiente virtual.

## 2. Abrir a pasta do projeto

No PowerShell, acesse a pasta em que você salvou o código. Por exemplo:

```powershell
cd "$HOME\Desktop\TinyML"
Get-ChildItem
```

A pasta deve conter, no mínimo:

```text
TinyML/
├── teste_preliminar.py
├── requirements.txt
├── explorar_cwru.py       (opcional, para uso futuro)
└── README.md
```

## 3. Criar o ambiente virtual

```powershell
py -m venv .venv
```

**Opção A: ativar o ambiente virtual**

```powershell
.\.venv\Scripts\Activate.ps1
```

Se houver bloqueio de execução de scripts, utilize **somente nessa sessão do PowerShell**:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Opção B: sem ativar o ambiente virtual (alternativa recomendada em máquinas com restrições)**

Você pode executar o Python do ambiente diretamente, sem mudar as políticas do PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe teste_preliminar.py
```

As próximas instruções usam a **Opção A**, com o ambiente ativado.

## 4. Instalar as dependências

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

O arquivo `requirements.txt` inclui **NumPy**, **SciPy**, **scikit-learn** e **PyTorch**.

## 5. Executar o teste

```powershell
python teste_preliminar.py
```

O script treina uma CNN original com **16 e 32 filtros**, constrói uma versão reduzida com **8 e 16 filtros** e calcula o **Macro-F1** das duas sobre os mesmos sinais artificiais de teste.

Ao terminar, o programa imprime o resumo em JSON e salva, **na pasta do script**:

- `resultados_teste.json` — métricas de referência e da rede reduzida;
- `pesos_referencia.pt` — pesos da rede original;
- `pesos_podado.pt` — pesos da rede reduzida.

Os arquivos `.pt` **não são executáveis para microcontrolador**.

## 6. Consultar os resultados

```powershell
Get-Content .\resultados_teste.json
Get-Item .\pesos_referencia.pt, .\pesos_podado.pt, .\resultados_teste.json | Select-Object Name, Length
```

Uma execução anterior com semente 42 produziu os números abaixo, **apenas com dados sintéticos**:

| Métrica | Original | Reduzida |
|---|---:|---:|
| Parâmetros | 2.852 | 788 |
| Arquivo de pesos PyTorch | 14.529 bytes | 6.161 bytes |
| Macro-F1 no teste | 1,00 | 1,00 |

A redução de parâmetros foi de **72,37%** e a do arquivo de pesos de **57,60%**. Os bytes dos arquivos podem variar conforme a versão do PyTorch. O desempenho de 1,00 **não garante** o mesmo resultado em sinais reais de vibração.

## 7. Exploração futura dos arquivos CWRU (opcional)

O script `explorar_cwru.py` pode produzir um inventário de arquivos `.mat` que forem obtidos manualmente da [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file). Ele não faz download automático nem treina a CNN.

Após organizar os arquivos na pasta `dados_cwru`, use:

```powershell
python explorar_cwru.py --dados dados_cwru
```

Antes de utilizar os registros experimentalmente, é necessário conferir nomes, cargas e rótulos no catálogo oficial da CWRU.

## 8. Problemas comuns

**`py` não é reconhecido:** instale o Python e habilite seu inicializador, ou use `python` no lugar de `py`.

**`pip` não é reconhecido:** execute `python -m pip ...` em vez de `pip ...`.

**Ativação bloqueada pelo PowerShell:** use a Opção B da seção 3 ou o comando `Set-ExecutionPolicy -Scope Process ...`, sem alterar permanentemente a política de segurança.

**`No module named torch` (ou `numpy`, `scipy`, `sklearn`):** verifique que o ambiente `.venv` está ativado e execute novamente `python -m pip install -r requirements.txt`.

**O JSON não apareceu:** verifique se o script terminou sem erros e procure em sua própria pasta: `Get-ChildItem .\resultados_teste.json`.

## Próximas etapas

O protocolo completo ainda prevê testes com dados reais da CWRU, novos níveis de compactação, quantização INT8 e medição de Flash, SRAM e latência no Raspberry Pi Pico.

**Autores:** Eduardo da Silva e José Vinicius Grecco.

