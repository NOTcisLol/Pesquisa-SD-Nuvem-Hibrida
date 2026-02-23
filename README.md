# Pesquisa: Arquitetura Distribuida para Stable Diffusion em Nuvem Hibrida

Artigo de pesquisa sobre a implementacao de uma arquitetura distribuida hibrida para treinamento (LoRA fine-tuning) e inferencia de modelos Stable Diffusion, utilizando Google Colab como provedor de GPU e maquinas locais como repositorios de modelos.

## Estrutura do Repositorio

```
.
├── README.md                          # Este arquivo
├── artigo/
│   └── artigo_pesquisa.md             # Artigo completo (IEEE/ABNT)
├── Geracao/
│   └── SD_Next_Colab.ipynb            # Notebook de inferencia (SD.NEXT)
└── Trteinamento/
    └── Treinamento_SD_Colab.ipynb     # Notebook de treinamento (LoRA)
```

## Resumo

Este trabalho propoe e implementa uma arquitetura distribuida que combina:

- **Google Colab** (GPU Tesla T4) para treinamento e inferencia
- **Maquina local** como servidor de modelos via Cloudflare Tunnel
- **Google Drive** como camada de persistencia e cache
- **LoRA (Low-Rank Adaptation)** para fine-tuning eficiente

## Temas Abordados

- Sistemas Distribuidos (arquitetura cliente-servidor, cache hierarquico, tolerancia a falhas)
- Processamento em Nuvem (IaaS, nuvem hibrida, ambientes efemeros)
- Modelos de Difusao Latente (Stable Diffusion, Dreambooth, LoRA)
- Gerenciamento de Recursos (GPU, VRAM, armazenamento distribuido)

## Formato

- Estrutura: **IEEE Conference Paper**
- Referencias: **ABNT NBR 6023:2018**

## Tecnologias

PyTorch | Hugging Face Diffusers | SD.NEXT | Google Colab | Cloudflare Tunnel | Google Drive | PEFT/LoRA | bitsandbytes

## Autor

Cairo Alberto

## Licenca

Este trabalho e disponibilizado para fins academicos e de pesquisa.
