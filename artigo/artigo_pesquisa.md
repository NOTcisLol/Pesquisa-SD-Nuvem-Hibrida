# Arquitetura Distribuida para Treinamento e Inferencia de Modelos de Difusao Estavel em Ambiente de Nuvem Hibrida

**Distributed Architecture for Stable Diffusion Model Training and Inference in a Hybrid Cloud Environment**

---

**Autor:** Cairo Connerton
**Instituicao:** PUC Goiàs
**Data:** Fevereiro de 2026
**Tipo:** Artigo de Pesquisa
**Formato:** IEEE Conference Paper / Referencias ABNT NBR 6023

---

## Resumo

Este artigo apresenta uma arquitetura distribuida hibrida para treinamento e inferencia de modelos generativos de imagem baseados em Stable Diffusion, utilizando Google Colab como provedor de GPU em nuvem e maquinas locais como repositorios de modelos. A solucao proposta emprega tunelamento via Cloudflare para comunicacao entre os nos, Google Drive como camada de persistencia, e tecnicas de fine-tuning LoRA (Low-Rank Adaptation) para treinamento eficiente. Sao discutidos os desafios de gerenciamento de recursos em ambientes efemeros, estrategias de cache e checkpoint, e a orquestracao de dependencias em um sistema distribuido heterogeneo. Os resultados demonstram a viabilidade de pipelines de IA generativa em infraestruturas de custo zero, com implicacoes para democratizacao do acesso a computacao de alto desempenho.

**Palavras-chave:** Sistemas Distribuidos, Computacao em Nuvem, Stable Diffusion, LoRA, Google Colab, Arquitetura Hibrida, Processamento GPU.

## Abstract

This paper presents a hybrid distributed architecture for training and inference of Stable Diffusion-based generative image models, using Google Colab as a cloud GPU provider and local machines as model repositories. The proposed solution employs Cloudflare tunneling for inter-node communication, Google Drive as a persistence layer, and LoRA (Low-Rank Adaptation) fine-tuning techniques for efficient training. We discuss resource management challenges in ephemeral environments, cache and checkpoint strategies, and dependency orchestration in a heterogeneous distributed system. Results demonstrate the feasibility of generative AI pipelines on zero-cost infrastructure, with implications for democratizing access to high-performance computing.

**Keywords:** Distributed Systems, Cloud Computing, Stable Diffusion, LoRA, Google Colab, Hybrid Architecture, GPU Processing.

---

## I. Introducao

A geracao de imagens por inteligencia artificial tem experimentado avancos significativos desde a publicacao dos modelos de difusao latente (Latent Diffusion Models - LDM) por Rombach et al. [1]. O Stable Diffusion, derivado dessa pesquisa, tornou-se um dos modelos generativos mais utilizados, permitindo a criacao de imagens de alta qualidade a partir de descricoes textuais (text-to-image). Entretanto, tanto o treinamento quanto a inferencia desses modelos demandam recursos computacionais significativos -- particularmente unidades de processamento grafico (GPUs) com grande capacidade de memoria VRAM.

Este cenario cria uma barreira de acesso para pesquisadores, artistas e desenvolvedores que nao dispoe de hardware dedicado. Plataformas de computacao em nuvem como o Google Colab oferecem acesso gratuito a GPUs (tipicamente NVIDIA Tesla T4 com 15 GB de VRAM), porem impoe restricoes severas: sessoes efemeras com tempo limitado, armazenamento volatil, e largura de banda restrita.

O presente trabalho propoe e implementa uma **arquitetura distribuida hibrida** que supera essas limitacoes ao combinar:

1. **Computacao GPU em nuvem** (Google Colab) para processamento pesado de treinamento e inferencia;
2. **Armazenamento local** (maquina do usuario) como repositorio primario de modelos;
3. **Persistencia em nuvem** (Google Drive) como camada de cache e checkpoint;
4. **Tunelamento de rede** (Cloudflare Tunnel) para comunicacao segura entre os nos.

A pesquisa e estruturada em duas frentes de implementacao: (a) um pipeline de treinamento fine-tuning utilizando LoRA via a biblioteca Diffusers da Hugging Face, e (b) uma plataforma de inferencia baseada no SD.NEXT com sistema de gerenciamento de modelos distribuido.

## II. Fundamentacao Teorica

### A. Modelos de Difusao Latente

Os modelos de difusao latente operam em um espaco latente comprimido, utilizando um autoencoder variacional (VAE) para codificar imagens em representacoes de menor dimensionalidade. O processo de difusao ocorre nesse espaco reduzido, diminuindo drasticamente o custo computacional em comparacao com modelos de difusao que operam diretamente no espaco de pixels [1].

O Stable Diffusion utiliza uma arquitetura U-Net condicionada por embeddings textuais produzidos por um encoder CLIP (Contrastive Language-Image Pre-Training). A geracao ocorre por um processo iterativo de denoising, partindo de ruido gaussiano puro ate uma imagem coerente.

### B. Fine-Tuning com LoRA

Low-Rank Adaptation (LoRA) [2] e uma tecnica de fine-tuning eficiente que congela os pesos do modelo pre-treinado e injeta matrizes de baixo posto (low-rank) treinaveis nas camadas de atencao. Para uma camada com peso W ∈ R^(d×k), o LoRA adiciona:

```
W' = W + BA
```

onde B ∈ R^(d×r) e A ∈ R^(r×k), com r << min(d, k). Isso reduz drasticamente o numero de parametros treinaveis (tipicamente de bilhoes para milhoes), possibilitando fine-tuning em GPUs com recursos limitados como a Tesla T4.

### C. Computacao em Nuvem e Sistemas Distribuidos

Segundo Tanenbaum e Van Steen [3], um sistema distribuido e "uma colecao de computadores independentes que se apresenta ao usuario como um sistema unico e coerente." A arquitetura proposta neste trabalho exemplifica esse conceito ao distribuir funcionalidades entre nos heterogeneos:

- **No de Computacao** (Google Colab): Provisionamento elastico de GPU;
- **No de Armazenamento** (Maquina Local): Repositorio persistente de modelos;
- **No de Persistencia** (Google Drive): Cache de estado e checkpoint.

O modelo de computacao em nuvem empregado classifica-se como **IaaS (Infrastructure as a Service)** no caso do Google Colab, com caracteristicas de **nuvem hibrida** ao integrar recursos locais e remotos [4].

### D. Tunelamento e Comunicacao entre Nos

O Cloudflare Tunnel (cloudflared) estabelece conexoes seguras entre servicos locais e a internet publica sem necessidade de configuracao de NAT, firewall ou IP estatico. O protocolo opera via QUIC/HTTP2, criando um tunel reverso que expoe endpoints locais atraves de URLs publicas temporarias (*.trycloudflare.com) [5].

## III. Arquitetura do Sistema

### A. Visao Geral

A arquitetura proposta segue o paradigma **cliente-servidor distribuido** com tres camadas:

```
+------------------------------------------------------------------+
|                    ARQUITETURA DISTRIBUIDA                         |
+------------------------------------------------------------------+
|                                                                    |
|  [MAQUINA LOCAL]          [INTERNET]          [GOOGLE COLAB]      |
|  +-----------------+     +----------+     +--------------------+  |
|  | Servidor HTTP   |<--->| Cloudflare|<--->| Cliente HTTP      |  |
|  | de Modelos      |     | Tunnel   |     | (ModelManager)    |  |
|  |                 |     | (QUIC)   |     |                   |  |
|  | G:\Models\      |     +----------+     | /content/sdnext/  |  |
|  | - Checkpoints   |                      | - SD.NEXT WebUI   |  |
|  | - LoRAs         |                      | - GPU Tesla T4    |  |
|  | - VAEs          |                      | - 15GB VRAM       |  |
|  | - Embeddings    |                      |                   |  |
|  +-----------------+                      +--------+-----------+  |
|                                                    |              |
|                                           +--------v-----------+  |
|                                           | Google Drive       |  |
|                                           | (Persistencia)     |  |
|                                           | - SD.NEXT clone    |  |
|                                           | - venv (deps)      |  |
|                                           | - Modelos cache    |  |
|                                           | - Configs          |  |
|                                           | - Checkpoints      |  |
|                                           +--------------------+  |
+------------------------------------------------------------------+
```

**Figura 1.** Diagrama da arquitetura distribuida hibrida proposta.

### B. Componente 1: Servidor Local de Modelos

O servidor local e implementado como um servico HTTP leve (Python) que expoe uma API REST para:

- `GET /api/health` - Verificacao de disponibilidade (health check);
- `GET /api/models` - Listagem de todos os modelos por categoria;
- `GET /api/models/{category}` - Listagem de modelos em categoria especifica;
- `GET /download/{category}/{filename}` - Download de modelo com suporte a Range requests (download resumivel);
- `POST /upload/{path}` - Upload de imagens geradas de volta para a maquina local.

O servidor e exposto publicamente via Cloudflare Tunnel (`start_server.bat`), gerando uma URL temporaria unica por sessao.

### C. Componente 2: Pipeline de Computacao em Nuvem (Google Colab)

O ambiente Colab executa duas funcoes principais:

**Pipeline de Treinamento (Notebook 1):**
Implementa fine-tuning Dreambooth-LoRA utilizando a biblioteca Diffusers da Hugging Face. O pipeline inclui:

1. Limpeza de disco (remocion de residuos de sessoes anteriores);
2. Diagnostico de hardware (verificacao de GPU via `nvidia-smi`);
3. Instalacao de dependencias (Diffusers, Accelerate, Transformers, PEFT, bitsandbytes);
4. Preparacao de dataset (extracao e normalizacao de imagens);
5. Treinamento com parametros otimizados para T4 (fp16, 8-bit Adam, batch size 1).

**Pipeline de Inferencia (Notebook 2 - SD.NEXT):**
Implementa uma plataforma completa de geracao de imagens com:

1. Setup de ambiente com versionamento rigido de dependencias;
2. Integracao com Google Drive para persistencia;
3. Sistema de gerenciamento de modelos (`ModelManager`) com download sob demanda;
4. Servidor WebUI acessivel via tunnel Cloudflare;
5. Sistema de checkpoint/restore para sessoes subsequentes.

### D. Componente 3: Camada de Persistencia (Google Drive)

O Google Drive atua como camada de persistencia, armazenando:

- Clone completo do SD.NEXT (via symlink `/content/sdnext -> /content/drive/MyDrive/SD_Data/sdnext`);
- Ambiente virtual Python (venv) com todas as dependencias;
- Modelos baixados (checkpoints, LoRAs, VAEs, embeddings);
- Configuracoes do sistema (config.json, ui-config.json);
- Manifesto de checkpoint com metadados de inventario.

A estrategia de symlinks permite que o SD.NEXT opere transparentemente a partir do Drive, sem necessidade de copias entre volumes.

## IV. Implementacao

### A. Gerenciamento de Dependencias

Um dos desafios criticos em ambientes Colab e a incompatibilidade de versoes de bibliotecas. O sistema implementa um gerenciador de dependencias que:

1. **Verifica versoes instaladas** contra um manifesto de compatibilidade;
2. **Sincroniza pacotes** com versoes especificas testadas (e.g., Gradio 3.43.2, Transformers 4.57.5);
3. **Aplica patches de compatibilidade** apos a instalacao;
4. **Persiste a lista de dependencias** no Google Drive (`requirements_colab.txt`) para restauracao rapida.

```python
packages_to_fix = [
    ("gradio", "3.43.2"),
    ("fastapi", "0.124.4"),
    ("transformers", "4.57.5"),
    ("huggingface_hub", "0.36.0"),
    # ... 17 pacotes com versoes fixas
]
```

Na implementacao, 17 pacotes precisaram de versionamento explicito, totalizando 719 dependencias no ambiente final. Este controle granular e essencial porque o Colab atualiza suas imagens base periodicamente, quebrando compatibilidade com o SD.NEXT.

### B. Sistema de Download Resumivel

O `ModelManager` implementa downloads com suporte a HTTP Range requests, permitindo retomada apos interrupcoes de rede:

```python
headers = {}
if temp_path.exists():
    existing_size = temp_path.stat().st_size
    headers['Range'] = f'bytes={existing_size}-'
    mode = 'ab'  # Append mode
```

O sistema inclui:
- Barra de progresso em tempo real com velocidade e ETA;
- Arquivos temporarios (`.downloading`) para prevenir corrupcao;
- Rename atomico apos conclusao;
- Download de metadados associados (JSON, previews, thumbnails).

### C. Mecanismo de Checkpoint/Restore

Para mitigar a efemeridade das sessoes Colab, o sistema implementa um mecanismo de checkpoint que persiste o estado completo no Google Drive:

1. **Checkpoint (salvar):** Serializa dependencias pip, inventaria modelos, salva configs e gera um manifesto JSON;
2. **Quick Resume (restaurar):** Recria symlinks, reinstala dependencias do checkpoint, verifica integridade do cache.

O manifesto JSON registra:
```json
{
  "timestamp": "2026-02-17T14:30:00",
  "total_models": 41,
  "total_models_size_gb": 12.3,
  "has_venv": true,
  "models_inventory": { ... }
}
```

A restauracao completa leva aproximadamente 1-2 minutos, comparado a 15-20 minutos do setup inicial.

### D. Parametros de Treinamento LoRA

O pipeline de treinamento utiliza a seguinte configuracao otimizada para GPU Tesla T4:

| Parametro | Valor | Justificativa |
|-----------|-------|---------------|
| Modelo Base | Stable Diffusion v1.5 | Melhor suporte a LoRA rapido |
| Resolucao | 512x512 | Limite de VRAM da T4 |
| Batch Size | 1 | Restricao de memoria |
| Gradient Accumulation | 1 | Trade-off velocidade/estabilidade |
| Learning Rate | 1e-4 | Padrao para LoRA |
| LR Scheduler | Constant | Simplificacao para treinos curtos |
| Max Steps | 1000 | Suficiente para LoRA de identidade |
| Precisao | FP16 (mixed) | Reducao de 50% no uso de VRAM |
| Otimizador | 8-bit Adam | Reducao adicional de memoria via bitsandbytes |
| Checkpointing | A cada 500 steps | Tolerancia a falhas |

**Tabela I.** Parametros de treinamento Dreambooth-LoRA otimizados para Tesla T4.

## V. Sistemas Distribuidos: Analise Aprofundada

### A. Classificacao da Arquitetura

A arquitetura proposta pode ser classificada segundo multiplas taxonomias de sistemas distribuidos:

**Modelo de Coulouris et al. [6]:**
- **Tipo:** Sistema distribuido heterogeneo;
- **Comunicacao:** Sincrona (HTTP request-response) via tunnel;
- **Acoplamento:** Fraco (nos independentes, sem estado compartilhado em memoria);
- **Transparencia:** Parcial -- transparencia de localizacao (modelos acessiveis por nome) e acesso (API uniforme), mas sem transparencia de falha (interrupcao de tunnel e visivel).

**Padrao Arquitetural:**
A solucao combina tres padroes classicos de sistemas distribuidos:

1. **Cliente-Servidor:** O Colab (cliente) consome modelos do servidor local;
2. **Cache Distribuido:** O Google Drive atua como cache L2, reduzindo latencia em sessoes subsequentes;
3. **Proxy Reverso:** O Cloudflare Tunnel atua como proxy, abstraindo a topologia de rede.

### B. Consistencia e Disponibilidade

Analisando pelo prisma do **Teorema CAP** [7]:

- **Consistencia (C):** Eventual -- modelos no cache podem divergir da maquina local se atualizados;
- **Disponibilidade (A):** Parcial -- depende do tunnel e da sessao Colab;
- **Tolerancia a Particionamento (P):** O sistema degrada graciosamente: se o tunnel cai, utiliza o cache do Drive.

O sistema prioriza **AP (Disponibilidade + Particionamento)** em detrimento de consistencia forte, o que e adequado para workloads de IA generativa onde versoes de modelos mudam com pouca frequencia.

### C. Gerenciamento de Recursos

O gerenciamento de recursos segue estrategias de sistemas distribuidos:

**Alocacao de Armazenamento:**
```
+-----------------+--------+----------+------------+
| Recurso         | Local  | Colab    | Drive      |
+-----------------+--------+----------+------------+
| Modelos (75 GB) | Master | Cache    | Cache L2   |
| GPU (T4 15GB)   | -      | Compute  | -          |
| Disco (112 GB)  | -      | Efemero  | Persistente|
| venv (deps)     | -      | Runtime  | Persistente|
+-----------------+--------+----------+------------+
```

**Tabela II.** Distribuicao de recursos entre nos do sistema.

**Politica de Cache:**
- **Write-through:** Modelos baixados sao escritos diretamente no Drive (via symlink);
- **Eviction manual:** Usuario decide quais modelos remover (`mm.delete()`, `mm.cleanup()`);
- **Prefetch seletivo:** Apenas modelos solicitados sao transferidos (download sob demanda).

### D. Tolerancia a Falhas

O sistema implementa multiplas estrategias de tolerancia a falhas:

1. **Downloads resumiveis:** Arquivos `.downloading` permitem retomada apos queda de conexao;
2. **Checkpoint no Drive:** Estado completo salvo para recuperacao de sessao;
3. **Quick Resume:** Restauracao em 1-2 min vs 15-20 min do setup completo;
4. **Limpeza de residuos:** `deep_cleanup()` remove artefatos de sessoes falhas anteriores;
5. **Fallback de scripts:** Se o script de treinamento nao e encontrado localmente, e baixado do GitHub;
6. **Instalacao com fallback:** Se uma versao especifica de pacote falha, tenta a versao mais recente.

### E. Escalabilidade

**Escalabilidade Vertical:** Limitada pelo tier do Google Colab (T4 no free tier, V100/A100 no Colab Pro).

**Escalabilidade Horizontal:** O servidor local pode atender multiplas sessoes Colab simultaneamente, pois o Cloudflare Tunnel suporta multiplexacao de conexoes. Entretanto, a largura de banda do tunnel limita o throughput efetivo (observado: ~4.4 MB/s para downloads de modelos de 6.5 GB).

## VI. Processamento em Nuvem: Analise Aprofundada

### A. Modelo de Execucao

O Google Colab oferece um modelo de execucao **serverless com GPU**, onde:

- Sessoes sao **efemeras** (tempo maximo de 12h no free tier);
- O armazenamento local (`/content/`) e **volatil** entre sessoes;
- GPUs sao alocadas **sob demanda** e podem ser revogadas;
- A conectividade de rede e fornecida pelo Google Cloud Platform.

Este modelo impoe desafios unicos para aplicacoes de IA generativa que requerem modelos grandes (2-7 GB cada) e ambientes complexos (719+ dependencias pip).

### B. Estrategia de Custo

A arquitetura proposta opera inteiramente no tier gratuito:

| Servico | Custo | Limite |
|---------|-------|--------|
| Google Colab (Free) | US$ 0 | ~12h/sessao, GPU T4 |
| Google Drive (Free) | US$ 0 | 15 GB |
| Cloudflare Tunnel | US$ 0 | Ilimitado |
| Maquina Local | Eletricidade | Armazenamento ilimitado |

**Tabela III.** Analise de custo da infraestrutura.

**Nota:** O Google Drive de 15 GB e insuficiente para armazenar muitos modelos. Na pratica, o usuario demonstrou uso de 54 GB (plano expandido), o que sugere necessidade de Google One para uso produtivo.

### C. Latencia e Desempenho

A latencia do sistema e dominada pela transferencia de modelos:

- **Download de checkpoint (6.5 GB):** ~25 minutos a 4.4 MB/s;
- **Download de LoRA (218 MB):** ~50 segundos a 4.4 MB/s;
- **Quick Resume (sem download):** ~1-2 minutos;
- **Treinamento LoRA (1000 steps):** Dependente do dataset (245 imagens ~ 30-60 min na T4);
- **Inferencia (geracao):** ~5-15 segundos por imagem (512x512, 20-30 steps).

A principal observacao e que o **gargalo nao e a GPU, mas a rede** -- a transferencia de modelos domina o tempo total de sessao. A estrategia de cache no Drive mitiga isso para sessoes subsequentes.

### D. Seguranca

O modelo de seguranca apresenta consideracoes importantes:

- **Tunelamento:** URLs temporarias do Cloudflare sao imprevisiveis, mas publicamente acessiveis. Nao ha autenticacao no servidor de modelos;
- **Google Drive:** Dados persistidos estao vinculados a conta Google do usuario;
- **Colab Runtime:** Codigo executa em container gerenciado pelo Google, com isolamento de processo;
- **Modelos:** Arquivos `.safetensors` incluem verificacao de integridade contra execucao de codigo arbitrario [8].

**Recomendacao:** Implementar autenticacao baseada em token no servidor de modelos para ambientes de producao.

## VII. Tecnologias Utilizadas

### A. Stack de Software

| Camada | Tecnologia | Versao | Papel |
|--------|------------|--------|-------|
| Modelo Base | Stable Diffusion | v1.5 / XL | Geracao e treinamento |
| Framework ML | PyTorch | (via Colab) | Runtime de redes neurais |
| Fine-tuning | Hugging Face Diffusers | Source (main) | Pipeline Dreambooth-LoRA |
| Otimizacao | PEFT, bitsandbytes | Latest | LoRA + quantizacao 8-bit |
| Aceleracao | Accelerate | Latest | Distribuicao de treino |
| WebUI | SD.NEXT (vladmandic) | Latest | Interface de inferencia |
| UI Framework | Gradio | 3.43.2 | WebUI components |
| Persistencia | Google Drive API | Colab SDK | Montagem e symlinks |
| Rede | Cloudflare Tunnel | Latest | Tunelamento reverso |
| Servidor | Python HTTP | Custom | API de modelos |
| Container | Google Colab | Free Tier | GPU + Runtime |

**Tabela IV.** Stack tecnologico completo do sistema.

### B. Hardware

| Recurso | Especificacao |
|---------|---------------|
| GPU (Colab) | NVIDIA Tesla T4, 15 GB VRAM, driver 580.82 |
| Disco (Colab) | 112 GB total, ~70 GB livres |
| RAM (Colab) | ~12 GB |
| Armazenamento Local | Variavel (observado: 75+ GB em modelos) |

**Tabela V.** Recursos de hardware utilizados.

## VIII. Desafios e Solucoes

### Desafio 1: Efemeridade das Sessoes Colab

**Problema:** Sessoes Colab sao destruidas apos inatividade ou tempo maximo, perdendo todo o estado local.

**Solucao:** Sistema de checkpoint/restore com Google Drive como backing store. Symlinks permitem operacao transparente diretamente do Drive. O `Quick Resume` restaura o ambiente completo em 1-2 minutos.

### Desafio 2: Incompatibilidade de Dependencias

**Problema:** O Colab atualiza suas imagens base periodicamente, quebrando compatibilidade. Na execucao observada, 17 de 17 pacotes criticos estavam em versoes incompativeis.

**Solucao:** Manifesto de versoes fixas com sincronizacao automatica. O sistema verifica, desinstala versoes incorretas e instala as testadas. Dependencias pip sao persistidas no Drive para restauracao rapida.

### Desafio 3: Limitacao de Armazenamento

**Problema:** Modelos de IA sao grandes (2-7 GB cada). O Colab oferece 112 GB de disco efemero e o Drive gratuito 15 GB.

**Solucao:** Arquitetura de cache hierarquico: modelos permanecem na maquina local (master), sao transferidos sob demanda para o Colab (cache L1) e persistidos no Drive (cache L2). O `ModelManager` oferece operacoes de limpeza seletiva.

### Desafio 4: Conectividade entre Nos

**Problema:** A maquina local esta atras de NAT/firewall, inacessivel diretamente pela internet.

**Solucao:** Cloudflare Tunnel cria um tunel reverso, gerando uma URL publica temporaria. A URL e manual (copiada pelo usuario), o que e uma limitacao de usabilidade mas garante controle.

### Desafio 5: Restricoes de VRAM para Treinamento

**Problema:** A Tesla T4 tem 15 GB de VRAM, insuficiente para fine-tuning convencional de modelos de difusao.

**Solucao:** Combinacao de LoRA (reduz parametros treinaveis), precisao mista FP16 (reduz uso de VRAM em ~50%), otimizador 8-bit Adam (via bitsandbytes), e batch size 1 com gradient accumulation.

### Desafio 6: Limpeza de Residuos

**Problema:** Tentativas falhas de sessoes anteriores deixam residuos no disco (scripts Kohya, modelos duplicados, caches).

**Solucao:** Funcao `deep_cleanup()` executada antes de qualquer operacao, removendo diretorios conhecidos de residuos e limpando cache do pip.

### Desafio 7: Transferencia de Modelos Grandes

**Problema:** Downloads de 6.5 GB via tunnel podem falhar por timeout ou instabilidade de rede.

**Solucao:** Downloads resumiveis via HTTP Range requests. Arquivos parciais sao mantidos (`.downloading`) e retomados automaticamente. Taxa observada: ~4.4 MB/s, com download de 6.5 GB concluido em ~25 minutos.

## IX. Resultados

### A. Metricas de Desempenho

A implementacao foi testada com os seguintes resultados:

| Metrica | Valor |
|---------|-------|
| Tempo de setup inicial | ~20 minutos (clone + deps + modelo) |
| Tempo de Quick Resume | ~1-2 minutos |
| Modelos gerenciados | 41 arquivos, 12.3 GB total no cache |
| Taxa de download (tunnel) | ~4.4 MB/s |
| Pacotes pip no ambiente | 719 |
| Imagens de treino processadas | 245 |
| Treinamento LoRA (1000 steps) | Concluido com sucesso na T4 |

**Tabela VI.** Metricas de desempenho observadas.

### B. Viabilidade

O sistema demonstrou viabilidade para:

1. **Treinamento LoRA:** 245 imagens processadas, 1000 steps executados com sucesso, modelo .safetensors gerado;
2. **Inferencia via SD.NEXT:** WebUI funcional acessivel via tunnel, suportando multiplos checkpoints (13 disponiveis) e 52 LoRAs;
3. **Persistencia:** Checkpoint completo salvo no Drive, restauracao funcional verificada;
4. **Gerenciamento de modelos:** 41 modelos catalogados em 7 categorias, com download/delete sob demanda.

## X. Trabalhos Relacionados

Diversas abordagens foram propostas para execucao de modelos de IA generativa em ambientes de nuvem:

- **RunPod / Vast.ai:** Plataformas de GPU-as-a-Service pagas, sem integracao local nativa;
- **Paperspace Gradient:** Oferece persistencia nativa, mas com custo;
- **Kaggle Notebooks:** Similar ao Colab, mas com restricoes mais severas de GPU;
- **ComfyUI Colab Notebooks:** Focados em inferencia, sem arquitetura distribuida para modelos locais.

A contribuicao diferencial deste trabalho e a **integracao hibrida local-nuvem com tunelamento**, permitindo uso do repositorio de modelos do usuario sem upload previo para a nuvem.

## XI. Conclusao

Este trabalho apresentou uma arquitetura distribuida hibrida funcional para treinamento e inferencia de modelos Stable Diffusion, combinando recursos de nuvem gratuitos (Google Colab + Drive) com infraestrutura local (servidor de modelos). A solucao demonstrou:

1. **Viabilidade tecnica** de pipelines de IA generativa em infraestrutura de custo zero;
2. **Eficacia de estrategias de cache** (reduzindo tempo de setup de 20 min para 1-2 min);
3. **Robustez do sistema** com downloads resumiveis, checkpoint/restore e limpeza automatica;
4. **Aplicabilidade de conceitos de sistemas distribuidos** (cache hierarquico, tolerancia a falhas, comunicacao por tunelamento) em contextos de IA.

### Trabalhos Futuros

- Implementacao de autenticacao no servidor de modelos;
- Suporte a multiplos nos de computacao (multi-Colab);
- Sincronizacao automatica de modelos (push-based vs pull-based);
- Migracao para Kubernetes para ambientes de producao;
- Quantizacao de modelos para reducao de tamanho de transferencia.

## Referencias

[1] ROMBACH, R. et al. High-Resolution Image Synthesis with Latent Diffusion Models. In: **Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)**. New Orleans: IEEE, 2022. p. 10684-10695. DOI: 10.1109/CVPR52688.2022.01042.

[2] HU, E. J. et al. LoRA: Low-Rank Adaptation of Large Language Models. In: **International Conference on Learning Representations (ICLR)**. 2022. Disponivel em: https://arxiv.org/abs/2106.09685. Acesso em: 23 fev. 2026.

[3] TANENBAUM, A. S.; VAN STEEN, M. **Distributed Systems: Principles and Paradigms**. 3. ed. London: Pearson, 2017. 596 p. ISBN: 978-1530281756.

[4] MELL, P.; GRANCE, T. **The NIST Definition of Cloud Computing**. Gaithersburg: National Institute of Standards and Technology, 2011. (Special Publication 800-145). DOI: 10.6028/NIST.SP.800-145.

[5] CLOUDFLARE, Inc. Cloudflare Tunnel Documentation. 2025. Disponivel em: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/. Acesso em: 23 fev. 2026.

[6] COULOURIS, G. et al. **Distributed Systems: Concepts and Design**. 5. ed. Boston: Addison-Wesley, 2011. 1067 p. ISBN: 978-0132143011.

[7] BREWER, E. A. Towards Robust Distributed Systems. In: **Proceedings of the 19th Annual ACM Symposium on Principles of Distributed Computing (PODC)**. Portland: ACM, 2000. Keynote address.

[8] HUGGING FACE. Safetensors: A Simple, Safe Way to Store and Distribute Tensors. 2023. Disponivel em: https://huggingface.co/docs/safetensors/. Acesso em: 23 fev. 2026.

---

**Nota sobre formatacao:** Este documento segue a estrutura de artigo IEEE Conference Paper (secoes numeradas em romanos, tabelas e figuras referenciadas no texto) com referencias bibliograficas formatadas segundo a norma ABNT NBR 6023:2018 (sobrenome em maiusculas, **titulo em negrito**, dados complementares em sequencia padronizada).
