# Guia de Instalação - CA Audio Synthesizer

## Métodos de Instalação

### 1. Instalação Local (Recomendada para uso completo)

#### Pré-requisitos
- Python 3.8 ou superior
- Git
- 4GB RAM mínimo (8GB recomendado)
- Placa de som compatível

#### Passo a Passo

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/ca-audio-synthesizer.git
cd ca-audio-synthesizer

# 2. Crie ambiente virtual
python -m venv venv

# 3. Ative o ambiente virtual
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 4. Atualize pip
python -m pip install --upgrade pip

# 5. Instale dependências
pip install -r requirements.txt

# 6. Teste a instalação
python -c "import numpy, matplotlib, scipy, librosa; print('Dependências OK!')"
```

#### Verificação da Instalação

```bash
# Teste básico da aplicação desktop
python CA_SF_53_working_audio_descriptors_changers_analysis.py

# Teste da aplicação web
streamlit run streamlit_ca_synthesizer.py
```

### 2. Streamlit Cloud (Acesso Web Rápido)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sua-app.streamlit.app)

**Vantagens:**
- Sem necessidade de instalação
- Acesso via navegador
- Atualizações automáticas

**Limitações:**
- Funcionalidades reduzidas
- Sem exportação de vídeos
- Limitações de processamento

### 3. Docker (Containerizado)

```bash
# Build da imagem
docker build -t ca-synthesizer .

# Executar container
docker run -p 8501:8501 ca-synthesizer
```

### 4. Instalação via pip (Futuro)

```bash
# Planejado para versões futuras
pip install ca-audio-synthesizer
```

## Dependências Detalhadas

### Principais
- **numpy**: Computação numérica e arrays
- **matplotlib**: Visualização e plotting
- **scipy**: Processamento de sinais e filtros
- **librosa**: Análise de áudio e descritores espectrais
- **sounddevice**: Reprodução de áudio em tempo real
- **tkinter**: Interface gráfica desktop (incluído no Python)

### Processamento de Áudio
- **pyroomacoustics**: Simulação acústica para reverb
- **imageio**: Processamento de imagens e vídeos
- **scikit-learn**: PCA e análise de dados

### Web (Streamlit)
- **streamlit**: Framework web
- **plotly**: Gráficos interativos

### Opcionais
- **soundfile**: Formatos de áudio adicionais
- **pydub**: Manipulação de áudio
- **ffmpeg**: Codecs de vídeo (para exportação)

## Instalação por Sistema Operacional

### Windows

#### Via Anaconda (Recomendado)
```bash
# 1. Instale Anaconda/Miniconda
# 2. Abra Anaconda Prompt
conda create -n ca-synth python=3.9
conda activate ca-synth

# 3. Clone e instale
git clone https://github.com/seu-usuario/ca-audio-synthesizer.git
cd ca-audio-synthesizer
pip install -r requirements.txt

# 4. Instale dependências do sistema
conda install ffmpeg -c conda-forge  # Para exportação de vídeo
```

#### Problemas Comuns Windows
- **Erro tkinter**: Reinstale Python com "tcl/tk and IDLE" marcado
- **Erro sounddevice**: Instale Microsoft Visual C++ Redistributable
- **Erro librosa**: `pip install --upgrade setuptools wheel`

### macOS

```bash
# 1. Instale Homebrew (se não tiver)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Instale Python e dependências do sistema
brew install python@3.9 ffmpeg portaudio

# 3. Clone e instale
git clone https://github.com/seu-usuario/ca-audio-synthesizer.git
cd ca-audio-synthesizer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Problemas Comuns macOS
- **Erro portaudio**: `brew install portaudio`
- **Permissões de microfone**: Permita acesso em Configurações > Segurança
- **Erro tkinter**: Use Python da Homebrew em vez do sistema

### Linux (Ubuntu/Debian)

```bash
# 1. Instale dependências do sistema
sudo apt update
sudo apt install python3 python3-pip python3-venv git
sudo apt install python3-tk ffmpeg libportaudio2-dev

# 2. Clone e instale
git clone https://github.com/seu-usuario/ca-audio-synthesizer.git
cd ca-audio-synthesizer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Problemas Comuns Linux
- **Erro ALSA**: `sudo apt install libasound2-dev`
- **Erro tkinter**: `sudo apt install python3-tk`
- **Permissões áudio**: Adicione usuário ao grupo audio: `sudo usermod -a -G audio $USER`

### Arch Linux

```bash
# Dependências do sistema
sudo pacman -S python python-pip git tk ffmpeg portaudio

# Clone e instale
git clone https://github.com/seu-usuario/ca-audio-synthesizer.git
cd ca-audio-synthesizer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configurações Específicas

### Configuração de Áudio

#### Windows
```python
# Teste dispositivos disponíveis
import sounddevice as sd
print(sd.query_devices())

# Configure dispositivo padrão se necessário
sd.default.device = 'nome-do-dispositivo'
```

#### Linux (PulseAudio)
```bash
# Verifique dispositivos
pactl list short sinks

# Configure dispositivo padrão
export PULSE_DEVICE="nome-do-dispositivo"
```

### Configuração de Performance

#### Para Grids Grandes (>100x100)
```python
# No início do script principal
import os
os.environ['OMP_NUM_THREADS'] = '4'  # Ajuste conforme CPU

# Configurações de memória
import numpy as np
np.seterr(over='ignore')  # Ignora overflow warnings
```

#### Para Tempo Real
```python
# Configurações de buffer de áudio
BUFFER_SIZE = 1024  # Menor = menor latência
SAMPLE_RATE = 44100  # Padrão para qualidade CD
```

## Instalação para Desenvolvimento

### Dependências Adicionais
```bash
pip install -r requirements-dev.txt
```

### Ferramentas de Desenvolvimento
- **pytest**: Testes unitários
- **black**: Formatação de código
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Hooks de commit

### Configuração do Ambiente
```bash
# Instale hooks de pre-commit
pre-commit install

# Execute testes
pytest tests/

# Formate código
black .

# Verifique estilo
flake8 .
```

## Verificação da Instalação

### Script de Teste Completo
```python
#!/usr/bin/env python3
"""Script de verificação da instalação"""

def test_dependencies():
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy.signal
        import librosa
        import sounddevice as sd
        import tkinter as tk
        print("✓ Dependências principais OK")
    except ImportError as e:
        print(f"✗ Erro nas dependências: {e}")
        return False
    return True

def test_audio_system():
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print(f"✓ Sistema de áudio OK ({len(devices)} dispositivos)")
    except Exception as e:
        print(f"✗ Erro no sistema de áudio: {e}")
        return False
    return True

def test_ca_synthesis():
    try:
        import numpy as np
        from scipy.signal import convolve2d
        
        # Teste básico de CA
        grid = np.random.choice([0, 1], size=(10, 10))
        kernel = np.ones((3, 3))
        neighbors = convolve2d(grid, kernel, mode='same') - grid
        print("✓ Algoritmos CA funcionando")
    except Exception as e:
        print(f"✗ Erro nos algoritmos CA: {e}")
        return False
    return True

def test_gui():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # Não mostra janela
        root.destroy()
        print("✓ Interface gráfica OK")
    except Exception as e:
        print(f"✗ Erro na interface gráfica: {e}")
        return False
    return True

if __name__ == "__main__":
    print("=== Verificação da Instalação CA Audio Synthesizer ===\n")
    
    tests = [
        test_dependencies,
        test_audio_system,
        test_ca_synthesis,
        test_gui
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Resultado: {passed}/{len(tests)} testes passaram")
    
    if passed == len(tests):
        print("🎉 Instalação completa e funcional!")
    else:
        print("⚠️  Alguns problemas detectados. Verifique a documentação.")
```

### Teste Rápido
```bash
# Salve o script acima como test_installation.py
python test_installation.py
```

## Solução de Problemas

### Erro: "No module named..."
```bash
# Verifique se está no ambiente virtual correto
which python
pip list

# Reinstale dependência específica
pip install --force-reinstall nome-do-modulo
```

### Erro: "Permission denied"
```bash
# Linux/Mac: Verifique permissões
ls -la
chmod +x CA_SF_53_working_audio_descriptors_changers_analysis.py

# Windows: Execute como administrador se necessário
```

### Erro: "ALSA/Audio driver"
```bash
# Linux: Reinstale drivers de áudio
sudo apt install --reinstall alsa-base pulseaudio

# Reinicie serviços de áudio
pulseaudio -k
pulseaudio --start
```

### Performance Lenta
1. **Reduza tamanho da grid** (<50x50 para tempo real)
2. **Limite gerações** (<20 para interatividade)
3. **Use menos camadas** (<5 simultaneamente)
4. **Feche outros programas** que usem CPU/RAM

### Erro de Importação Streamlit
```bash
# Atualize streamlit
pip install --upgrade streamlit

# Limpe cache
streamlit cache clear
```

## Desinstalação

### Completa
```bash
# Remova ambiente virtual
rm -rf venv

# Remova diretório do projeto
cd ..
rm -rf ca-audio-synthesizer
```

### Apenas Dependências
```bash
# Desative ambiente
deactivate

# Remova ambiente virtual
rm -rf venv
```

## Suporte

Para problemas de instalação:

1. **Verifique FAQ**: [Link para FAQ]
2. **Issues GitHub**: Para bugs específicos
3. **Discussions**: Para dúvidas gerais
4. **Discord**: Chat em tempo real

### Informações para Suporte

Ao reportar problemas, inclua:
- Sistema operacional e versão
- Versão do Python (`python --version`)
- Lista de pacotes instalados (`pip list`)
- Mensagem de erro completa
- Passos para reproduzir

---

*Este guia será atualizado conforme necessário. Para versão mais recente, consulte o repositório oficial.*