# 🎵 Sintetizador de Áudio com Autômatos Celulares

Uma aplicação Python para síntese sonora e composição musical baseada em padrões de Autômatos Celulares (CA). Este projeto oferece duas interfaces distintas: uma aplicação desktop completa com Tkinter e uma versão web simplificada com Streamlit.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Características](#características)
- [Instalação](#instalação)
- [Uso](#uso)
- [Documentação](#documentação)
- [Exemplos](#exemplos)
- [Contribuição](#contribuição)
- [Licença](#licença)

## 🎯 Visão Geral

Este sintetizador converte padrões evolutivos de Autômatos Celulares em material sonoro, permitindo:

- **Síntese Granular**: Cada célula viva gera grãos sonoros
- **Mapeamento Espacial**: Coordenadas X/Y determinam frequências
- **Evolução Temporal**: Gerações do CA criam progressões temporais
- **Múltiplas Camadas**: Combinação de diferentes tipos de CA
- **Controle de Descritores**: Manipulação de características espectrais

### Tipos de Autômatos Celulares Suportados

- **Game of Life**: Regras clássicas de Conway
- **HighLife**: Extensão com replicação
- **Seeds**: Padrões explosivos simples
- **Day and Night**: Regras invertidas
- **Anneal, Bacteria, Maze, Coral**: Variações especializadas

## ✨ Características

### Versão Desktop (Tkinter)
- Interface multi-abas com controles detalhados
- Síntese em tempo real com múltiplas camadas
- Exportação de áudio, imagens, GIFs e vídeos
- Controle avançado de filtros e efeitos
- Sistema de descritores de áudio interativo
- Algoritmo genético para otimização de parâmetros

### Versão Web (Streamlit)
- Interface simplificada e intuitiva
- Visualização em tempo real
- Player de áudio integrado
- Análise espectral básica
- Exportação de áudio e animações

## 🛠 Instalação

### Requisitos do Sistema
- Python 3.8 ou superior
- Sistema operacional: Windows, macOS, ou Linux

### Dependências Principais

```bash
# Instalação das dependências
pip install -r requirements.txt
```

### Instalação por Método

#### 1. Aplicação Desktop

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/ca-audio-synthesizer.git
cd ca-audio-synthesizer

# Instale as dependências
pip install -r requirements.txt

# Execute a aplicação
python CA_SF_53_working_audio_descriptors_changers_analysis.py
```

#### 2. Aplicação Web (Local)

```bash
# Execute com Streamlit
streamlit run streamlit_ca_synthesizer.py
```

#### 3. Aplicação Web (Streamlit Cloud)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sua-app.streamlit.app)

## 🎮 Uso

### Versão Desktop

1. **Configuração de Camadas**:
   - Ajuste o número de camadas no painel "Layer Settings"
   - Configure cada camada individualmente

2. **Parâmetros do CA**:
   - Escolha o tipo de autômato celular
   - Defina dimensões da grade (largura/altura)
   - Ajuste densidade inicial e número de gerações

3. **Parâmetros de Áudio**:
   - Selecione forma de onda (Sine, Square, Sawtooth, etc.)
   - Configure faixas de frequência (min/max)
   - Aplique técnicas de síntese (AM, FM, Ring Modulation)

4. **Controles Globais**:
   - Gere novas iterações
   - Reproduza, pause ou pare o áudio
   - Exporte arquivos de áudio, imagens ou vídeos

### Versão Web

1. **Configuração na Barra Lateral**:
   - Selecione tipo de CA e parâmetros da grade
   - Ajuste parâmetros de áudio e filtros

2. **Geração**:
   - Clique em "Generate Audio" para criar o som
   - Visualize a evolução do CA no painel principal

3. **Reprodução e Análise**:
   - Use o player de áudio integrado
   - Analise as características espectrais
   - Baixe áudio e animações geradas

## 📚 Documentação

### Estrutura do Projeto

```
ca-audio-synthesizer/
├── CA_SF_53_working_audio_descriptors_changers_analysis.py  # Aplicação desktop
├── streamlit_ca_synthesizer.py                             # Aplicação web
├── requirements.txt                                         # Dependências
├── README.md                                               # Este arquivo
├── README_EN.md                                            # Versão em inglês
├── docs/                                                   # Documentação adicional
│   ├── user_guide.md                                      # Guia do usuário
│   ├── technical_details.md                               # Detalhes técnicos
│   └── examples.md                                        # Exemplos de uso
├── examples/                                              # Arquivos de exemplo
│   ├── audio_samples/                                     # Amostras de áudio
│   ├── ca_patterns/                                       # Padrões de CA
│   └── videos/                                            # Vídeos demonstrativos
└── assets/                                                # Recursos visuais
    ├── screenshots/                                       # Capturas de tela
    └── diagrams/                                          # Diagramas explicativos
```

### Algoritmo de Síntese

1. **Inicialização**: Grade aleatória baseada na densidade configurada
2. **Evolução**: Aplicação das regras do CA por N gerações
3. **Mapeamento**: Conversão de coordenadas em frequências
4. **Síntese**: Geração de grãos sonoros para cada célula viva
5. **Processamento**: Aplicação de filtros e efeitos
6. **Mixagem**: Combinação de múltiplas camadas

### Descritores de Áudio

- **Centroide Espectral**: Brilho timbral
- **Desvio Padrão Espectral**: Dispersão espectral
- **Assimetria Espectral**: Inclinação da distribuição
- **Curtose Espectral**: Concentração espectral
- **RMS**: Amplitude eficaz
- **Planicidade Espectral**: Características tonais/ruidosas

## 🎵 Exemplos

### Exemplo Básico (Game of Life)

```python
# Configuração simples para Game of Life
params = {
    'ca_type': 'Game of Life',
    'width': 50,
    'height': 50,
    'generations': 20,
    'density': 0.3,
    'duration': 3.0,
    'min_freq': 200,
    'max_freq': 1000,
    'waveform': 'Sine'
}
```

### Exemplo Avançado (Múltiplas Camadas)

```python
# Configuração multicamada com diferentes CAs
layer_1 = {
    'ca_type': 'Seeds',
    'synthesis': 'FM',
    'filter': 'lowpass'
}

layer_2 = {
    'ca_type': 'HighLife', 
    'synthesis': 'AM',
    'filter': 'highpass'
}
```

## 🔧 Parâmetros Avançados

### Técnicas de Síntese
- **Amplitude Modulation (AM)**: Modulação de amplitude
- **Frequency Modulation (FM)**: Modulação de frequência  
- **Ring Modulation**: Modulação em anel

### Filtros de Áudio
- **Low Pass**: Passa-baixa
- **High Pass**: Passa-alta
- **Band Pass**: Passa-faixa

### Mapeamentos de Frequência
- **Standard**: Mapeamento linear simples
- **Complex**: Baseado no número de vizinhos
- **Rule-Based**: Baseado nas regras do CA
- **Interpolation**: Interpolação entre vizinhos
- **Harmonics**: Séries harmônicas

## 🤝 Contribuição

Contribuições são bem-vindas! Por favor:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

### Áreas para Contribuição

- Novos tipos de autômatos celulares
- Algoritmos de síntese adicionais
- Melhorias na interface do usuário
- Otimizações de performance
- Documentação e exemplos
- Testes automatizados

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- Inspirado pelos trabalhos de John Conway (Game of Life)
- Baseado em conceitos de síntese granular
- Utiliza bibliotecas open-source da comunidade Python

## 📞 Contato

- **Autor**: [Seu Nome]
- **Email**: seu.email@exemplo.com
- **GitHub**: [@seu-usuario](https://github.com/seu-usuario)

---

*Criado com ❤️ para a comunidade de música eletrônica e programação criativa*