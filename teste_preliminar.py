import json
import math
import random
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
import torch
from torch import nn
from sklearn.metrics import f1_score

SEED = 42
N = 2048
FS = 48000
CLASSES = ("normal", "pista_interna", "esfera", "pista_externa")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)
OUT = Path(__file__).resolve().parent


def exemplo_sintetico(classe, carga, rng):
    """Sinais artificiais para testar o CODIGO, nao representar dados reais."""
    t = np.arange(N, dtype=np.float32) / FS
    # Padrões diferenciados por frequencia e modulação, com ruido e pequena mudança de carga.
    frequencias = (110, 290, 480, 750)
    freq = frequencias[classe] * (1 + .008 * carga) + rng.uniform(-4, 4)
    deslocamento = rng.uniform(0, 2 * np.pi)
    sinal = np.sin(2 * np.pi * freq * t + deslocamento)
    sinal += .34 * np.sin(2 * np.pi * (freq * 2.1) * t + deslocamento / 2)
    sinal += rng.normal(0, .23 + .02 * carga, N)
    return sinal.astype(np.float32)


def dados(carga, por_classe, rng):
    x = []
    y = []
    for c in range(len(CLASSES)):
        for _ in range(por_classe):
            x.append(exemplo_sintetico(c, carga, rng))
            y.append(c)
    return np.stack(x), np.asarray(y, dtype=np.int64)


class CNN(nn.Module):
    def __init__(self, c1=16, c2=32):
        super().__init__()
        self.conv1 = nn.Conv1d(1, c1, 7, padding=3)
        self.conv2 = nn.Conv1d(c1, c2, 5, padding=2)
        self.pool = nn.MaxPool1d(4)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return self.fc(self.avg(x).squeeze(-1))


def treinar(modelo, x, y, epocas, taxa):
    opt = torch.optim.Adam(modelo.parameters(), lr=taxa)
    indices = torch.arange(len(y))
    modelo.train()
    for _ in range(epocas):
        for ids in indices[torch.randperm(len(y))].split(32):
            opt.zero_grad()
            perda = nn.functional.cross_entropy(modelo(x[ids]), y[ids])
            perda.backward()
            opt.step()


def avaliar(modelo, x, y):
    modelo.eval()
    with torch.no_grad():
        pred = modelo(x).argmax(dim=1).numpy()
    return float(f1_score(y.numpy(), pred, average="macro", labels=list(range(4)), zero_division=0))


def podar_metade(modelo):
    """Poda estruturada por norma L1; remove fisicamente canais e filtros."""
    p = CNN(8, 16)
    with torch.no_grad():
        k1 = torch.topk(modelo.conv1.weight.abs().flatten(1).sum(1), k=8).indices.sort().values
        k2 = torch.topk(modelo.conv2.weight.abs().flatten(1).sum(1), k=16).indices.sort().values
        p.conv1.weight.copy_(modelo.conv1.weight[k1])
        p.conv1.bias.copy_(modelo.conv1.bias[k1])
        p.conv2.weight.copy_(modelo.conv2.weight[k2][:, k1])
        p.conv2.bias.copy_(modelo.conv2.bias[k2])
        p.fc.weight.copy_(modelo.fc.weight[:, k2])
        p.fc.bias.copy_(modelo.fc.bias)
    return p


def main():
    rng = np.random.default_rng(SEED)
    train0, y0 = dados(0, 48, rng)
    train1, y1 = dados(1, 48, rng)
    val, yv = dados(2, 24, rng)
    test, yt = dados(3, 24, rng)
    treino = np.vstack([train0, train1]); ytreino = np.concatenate([y0, y1])

    # Teste real de leitura de formato .mat, com conteudo ARTIFICIAL.
    mat_teste = OUT / '_temp_sintetico.mat'
    savemat(mat_teste, {'X_DE_time': treino[0].reshape(-1, 1)})
    mat_lido = loadmat(mat_teste)['X_DE_time'].reshape(-1)
    mat_ok = bool(np.array_equal(mat_lido, treino[0]))
    mat_teste.unlink()

    mu = float(treino.mean()); sigma = float(treino.std())
    if not sigma: raise RuntimeError('Desvio-padrao zero')
    def preparar(a):
        return torch.from_numpy(((a - mu) / sigma)[:, None, :].astype(np.float32))
    xtr, xval, xte = map(preparar, (treino, val, test))
    yytr, yyval, yyte = map(torch.from_numpy, (ytreino, yv, yt))

    original = CNN()
    treinar(original, xtr, yytr, epocas=9, taxa=0.003)
    f0_val = avaliar(original, xval, yyval)
    f0 = avaliar(original, xte, yyte)
    podado = podar_metade(original)
    treinar(podado, xtr, yytr, epocas=4, taxa=0.001)
    fp_val = avaliar(podado, xval, yyval)
    fp = avaliar(podado, xte, yyte)
    
    # Compara as respostas individuais das duas redes
    original.eval()
    podado.eval()

    with torch.no_grad():
        respostas_original = original(xte).argmax(dim=1).numpy()
        respostas_reduzida = podado(xte).argmax(dim=1).numpy()

    respostas_corretas = yyte.numpy()

    print("\nCOMPARACAO DAS RESPOSTAS")
    print("Sinal | Correta | Original | Reduzida")

    for i in range(20):
        print(
            f"{i+1:2d} | "
            f"{CLASSES[respostas_corretas[i]]:15s} | "
            f"{CLASSES[respostas_original[i]]:15s} | "
            f"{CLASSES[respostas_reduzida[i]]}"
        )

    diferencas = (respostas_original != respostas_reduzida).sum()

    print("\nRespostas diferentes:", diferencas)
    print("Acertos original:", (respostas_original == respostas_corretas).sum())
    print("Acertos reduzida:", (respostas_reduzida == respostas_corretas).sum())


    orig_path = OUT / 'pesos_referencia.pt'
    pod_path = OUT / 'pesos_podado.pt'
    torch.save(original.state_dict(), orig_path)
    torch.save(podado.state_dict(), pod_path)
    so, sp = orig_path.stat().st_size, pod_path.stat().st_size
    res = {
        'aviso': 'APENAS teste de funcionamento com dados sinteticos: nao sao medidas da CWRU nem do Pico; nao sao modelos TFLite',
        'seed': SEED, 'comprimento_janela': N, 'amostras_treino': int(len(ytreino)),
        'amostras_validacao': int(len(yv)), 'amostras_teste': int(len(yt)),
        'formato_mat_teste_leitura': mat_ok,
        'framework': 'PyTorch '+torch.__version__,
        'referencia': {'parametros': sum(z.numel() for z in original.parameters()), 'bytes_pesos_pt': so, 'macroF1_validacao': round(f0_val, 4), 'macroF1_teste': round(f0, 4)},
        'poda_metade': {'parametros': sum(z.numel() for z in podado.parameters()), 'bytes_pesos_pt': sp, 'macroF1_validacao': round(fp_val, 4), 'macroF1_teste': round(fp, 4)},
        'reducao_parametros_pct': round(100 * (1-sum(z.numel() for z in podado.parameters())/sum(z.numel() for z in original.parameters())), 2),
        'reducao_arquivos_pt_pct': round(100*(1-sp/so), 2),
        'faltam': ['dataset real CWRU', 'TensorFlow/Keras', 'conversao .tflite', 'quantizacao INT8', 'medicoes na placa Raspberry Pi Pico']
    }
    (OUT / 'resultados_teste.json').write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()