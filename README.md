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

## Autores

Eduardo da Silva e José Vinicius Grecco.
