# Exemplos de Uso - CA Audio Synthesizer

## Índice
1. [Exemplos Básicos](#exemplos-básicos)
2. [Técnicas Avançadas](#técnicas-avançadas)
3. [Casos de Uso Específicos](#casos-de-uso-específicos)
4. [Projetos Criativos](#projetos-criativos)
5. [Códigos de Exemplo](#códigos-de-exemplo)

## Exemplos Básicos

### 1. Primeira Composição Simples

**Objetivo**: Criar uma melodia orgânica usando Game of Life

**Configuração**:
- CA Type: Game of Life
- Grid: 40x40
- Densidade: 0.25
- Duração: 4 segundos
- Frequências: 220-880 Hz (uma oitava de Lá)
- Forma de onda: Sine

**Procedimento**:
1. Configure uma camada única com os parâmetros acima
2. Gere 15 gerações para evolução moderada
3. Use mapeamento de frequência padrão
4. Aplique Low Pass filter em 2000 Hz para suavizar

**Resultado Esperado**: Som melodioso e orgânico com padrões que evoluem naturalmente.

### 2. Paisagem Sonora Ambiental

**Objetivo**: Criar textura ambiente usando múltiplas camadas

**Configuração Multicamada**:

**Camada 1 (Base)**:
- CA: Game of Life
- Grid: 60x60, densidade 0.3
- Frequências: 80-200 Hz
- Waveform: Sine
- Duração: 8 segundos
- Filter: Low Pass 300 Hz

**Camada 2 (Meio)**:
- CA: HighLife
- Grid: 40x40, densidade 0.2
- Frequências: 400-1200 Hz
- Waveform: Triangle
- Duração: 6 segundos
- Filter: Band Pass 800 Hz

**Camada 3 (Agudos)**:
- CA: Seeds
- Grid: 30x30, densidade 0.1
- Frequências: 2000-4000 Hz
- Waveform: White Noise
- Duração: 3 segundos
- Filter: High Pass 1500 Hz

**Procedimento**:
1. Configure as três camadas com parâmetros diferentes
2. Ajuste volumes relativos (Base: 80%, Meio: 60%, Agudos: 40%)
3. Adicione reverb global (Wet: 0.4, Room Size: 70)
4. Gere e escute o resultado

**Resultado**: Paisagem sonora rica e evolutiva com profundidade espectral.

### 3. Ritmo Orgânico com Seeds

**Objetivo**: Criar padrões rítmicos usando Seeds CA

**Configuração**:
- CA Type: Seeds
- Grid: 25x25
- Densidade: 0.15
- Instâncias: 8
- Duração por instância: 0.5 segundos
- Time span: 0.2 segundos
- Frequências: 150-600 Hz
- Waveform: Square

**Procedimento**:
1. Configure Seeds com baixa densidade para explosões controladas
2. Use instâncias curtas para criar pulsos rítmicos
3. Aplique Band Pass filter (300-800 Hz) para foco espectral
4. Exporte e analise padrões rítmicos gerados

**Resultado**: Sequência rítmica orgânica com pulsos irregulares mas musicais.

## Técnicas Avançadas

### 4. Controle de Brilho Espectral

**Objetivo**: Controlar a evolução do brilho sonoro usando descritores

**Configuração Base**:
- CA: Game of Life
- Grid: 50x50, densidade 0.3
- Frequências: 200-3000 Hz
- Instâncias: 6
- Descriptor: Spectral Centroid

**Procedimento**:
1. Configure o CA básico
2. Selecione "Spectral Centroid" como descritor
3. No canvas interativo, desenhe uma curva crescente:
   - Instância 1: y = 0.2 (som escuro)
   - Instância 3: y = 0.5 (neutro)
   - Instância 6: y = 0.9 (som brilhante)
4. Gere e observe a evolução do brilho

**Análise**: O som começará escuro e gradualmente ficará mais brilhante, demonstrando controle preciso de características espectrais.

### 5. Síntese FM Complexa

**Objetivo**: Explorar modulação de frequência com padrões CA

**Configuração**:
- CA: HighLife
- Synthesis Technique: Frequency Modulation
- Carrier Frequency: 440 Hz
- FM Frequency: 2-20 Hz (variável por posição)
- Modulation Index: 5.0
- Grid: 35x35

**Procedimento**:
1. Configure HighLife como CA base
2. Ative FM synthesis
3. Varie FM frequency baseado na posição das células
4. Use alta modulation index para timbres complexos
5. Aplique filtros para moldar o resultado

**Resultado**: Timbres complexos e evolutivos com características espectrais ricas.

### 6. Harmonias Generativas

**Objetivo**: Criar harmonias usando múltiplas camadas com diferentes escalas

**Configuração Harmônica**:

**Camada 1 (Fundamental)**:
- Frequências: 110, 146.83, 164.81, 220 Hz (escala pentatônica de Lá)
- CA: Game of Life, grid 40x40

**Camada 2 (Quinta)**:
- Frequências: 165, 220.08, 247.23, 330 Hz
- CA: HighLife, grid 35x35

**Camada 3 (Oitava)**:
- Frequências: 220, 293.66, 329.63, 440 Hz
- CA: Game of Life, grid 30x30

**Procedimento**:
1. Configure cada camada com frequências harmônicas relacionadas
2. Use CAs diferentes para variação rítmica
3. Sincronize durações para concordância temporal
4. Ajuste volumes para equilíbrio harmônico

## Casos de Uso Específicos

### 7. Sonorização de Instalação Artística

**Contexto**: Som para instalação interativa em galeria

**Requisitos**:
- Evolução lenta e contínua
- Sem repetições óbvias
- Texturas ricas mas não agressivas
- Duração: loop de 10 minutos

**Solução**:
```
Configuração A (5 min):
- 4 camadas com Game of Life
- Grids grandes (80x80) para evolução lenta
- Frequências graves e médias
- Reverb alto para espacialização

Configuração B (5 min):
- Transição para HighLife
- Gradual aumento de densidade
- Introdução de elementos agudos
- Retorno suave à configuração A
```

### 8. Trilha para Performance Audiovisual

**Contexto**: Acompanhamento sonoro para projeções visuais sincronizadas

**Estratégia**:
1. **Sincronização**: Use os mesmos seeds de CA para áudio e visual
2. **Mapeamento**: Correlacione brilho visual com centroide espectral
3. **Dinâmica**: Varie densidade CA conforme intensidade visual
4. **Transições**: Use interpolação entre estados para mudanças suaves

**Implementação**:
```python
# Pseudocódigo para sincronização
visual_brightness = calculate_visual_brightness(frame)
audio_centroid_target = map_brightness_to_centroid(visual_brightness)
adjust_ca_parameters_for_target(audio_centroid_target)
```

### 9. Composição Algorítmica Interativa

**Contexto**: Sistema responsivo a input externo (MIDI, OSC, etc.)

**Arquitetura**:
```
Input Device → Parameter Mapper → CA Controller → Audio Generator
     ↓              ↓                ↓              ↓
   MIDI CC    →   Density    →   Grid State  →   Audio Out
   Velocity   →   Rule Type  →   Evolution   →   Mix Levels
   Pitch Bend →   Filter Q   →   Generations →   Effects
```

**Implementação**: Modifique o código principal para aceitar input externo e mapear para parâmetros CA em tempo real.

## Projetos Criativos

### 10. "Evolução Urbana" - Composição Conceitual

**Conceito**: Sonificar crescimento urbano usando CAs diferentes para representar fases históricas

**Movimento I - Fundação (Séc. XVII)**:
- Game of Life básico
- Frequências graves (60-200 Hz)
- Evolução lenta, padrões estáveis

**Movimento II - Industrialização (Séc. XIX)**:
- Seeds explosivos
- Frequências médias com ruído
- Ritmos acelerados

**Movimento III - Metrópole (Séc. XX)**:
- Múltiplas camadas HighLife
- Espectro completo
- Complexidade máxima

**Movimento IV - Smart City (Séc. XXI)**:
- Algoritmo genético para otimização
- Descritores controlados por dados
- Síntese híbrida AM/FM

### 11. "Jardim Digital" - Instalação Interativa

**Conceito**: Cada visitante "planta" uma semente CA que influencia o ecossistema sonoro

**Implementação**:
1. Interface touch permite desenhar padrão inicial
2. Padrão vira seed para nova camada CA
3. Camadas interagem através de crossover genético
4. Som evolui continuamente com novas contribuições

**Parâmetros Interativos**:
- Posição do toque → frequência base
- Velocidade do gesto → densidade inicial
- Tempo de permanência → número de gerações

### 12. "Memórias Celulares" - Álbum Generativo

**Conceito**: Álbum onde cada faixa explora um tipo diferente de CA

**Estrutura**:
1. **Genesis** (Game of Life) - Criação e ordem emergente
2. **Explosion** (Seeds) - Caos e expansão rápida
3. **Replication** (HighLife) - Cópias e variações
4. **Day/Night** (Day and Night) - Ciclos e inversões
5. **Crystalline** (Maze) - Estruturas cristalinas
6. **Organic** (Coral) - Crescimento orgânico
7. **Decay** (Anneal) - Degeneração e transformação
8. **Genesis Reprise** - Retorno com variações

## Códigos de Exemplo

### Script para Automação de Variações

```python
import itertools
import os

def generate_variations():
    """Gera múltiplas variações automaticamente"""
    ca_types = ['Game of Life', 'Seeds', 'HighLife']
    densities = [0.2, 0.3, 0.4]
    grid_sizes = [30, 40, 50]
    
    for ca_type, density, size in itertools.product(ca_types, densities, grid_sizes):
        config = {
            'ca_type': ca_type,
            'density': density,
            'grid_size': (size, size),
            'generations': 20,
            'duration': 3.0,
            'filename': f"{ca_type}_{density}_{size}"
        }
        
        generate_and_export(config)
        print(f"Generated: {config['filename']}")

def generate_and_export(config):
    """Gera áudio com configuração específica e exporta"""
    # Implemente a lógica de geração aqui
    pass
```

### Análise Estatística de Padrões CA

```python
import numpy as np
import matplotlib.pyplot as plt

def analyze_ca_statistics(ca_sequences):
    """Analisa estatísticas de sequências CA"""
    stats = {}
    
    for name, sequence in ca_sequences.items():
        # Densidade ao longo do tempo
        densities = [np.mean(gen) for gen in sequence]
        
        # Variabilidade
        variability = np.std(densities)
        
        # Tendência (regressão linear simples)
        x = np.arange(len(densities))
        trend = np.polyfit(x, densities, 1)[0]
        
        stats[name] = {
            'mean_density': np.mean(densities),
            'variability': variability,
            'trend': trend,
            'final_density': densities[-1]
        }
        
        # Plot evolução
        plt.figure(figsize=(10, 6))
        plt.plot(densities, label=name)
        plt.title(f'{name} - Evolução da Densidade')
        plt.xlabel('Geração')
        plt.ylabel('Densidade')
        plt.legend()
        plt.savefig(f'{name}_evolution.png')
        plt.close()
    
    return stats
```

### Exportador Batch

```python
def batch_export_project():
    """Exporta projeto completo em diferentes formatos"""
    layers = get_all_layers()
    base_filename = get_project_name()
    
    export_formats = {
        'audio': ['wav', 'mp3'],
        'video': ['mp4', 'gif'],
        'image': ['png', 'jpg'],
        'data': ['json', 'csv']
    }
    
    for layer_idx, layer in enumerate(layers):
        layer_name = f"{base_filename}_layer_{layer_idx+1}"
        
        # Áudio
        for fmt in export_formats['audio']:
            export_audio(layer, f"{layer_name}.{fmt}")
        
        # Vídeo
        for fmt in export_formats['video']:
            export_video(layer, f"{layer_name}.{fmt}")
        
        # Imagens (frames individuais)
        export_images(layer, f"{layer_name}_frame")
        
        # Dados (parâmetros e estatísticas)
        export_layer_data(layer, f"{layer_name}_data.json")
    
    # Arquivo master com todas as configurações
    export_project_config(f"{base_filename}_config.json")
    
    print(f"Projeto exportado: {len(layers)} camadas em {sum(len(formats) for formats in export_formats.values())} formatos")
```

### Sistema de Presets

```python
class PresetManager:
    def __init__(self):
        self.presets = {
            'ambient_drone': {
                'ca_type': 'Game of Life',
                'grid_size': (60, 60),
                'density': 0.25,
                'generations': 30,
                'frequency_range': (80, 400),
                'waveform': 'sine',
                'filter': {'type': 'lowpass', 'cutoff': 800},
                'reverb': {'enabled': True, 'wet': 0.6, 'room_size': 80}
            },
            
            'rhythmic_pulse': {
                'ca_type': 'Seeds',
                'grid_size': (30, 30),
                'density': 0.15,
                'generations': 15,
                'instance_duration': 0.3,
                'frequency_range': (200, 1000),
                'waveform': 'square',
                'filter': {'type': 'bandpass', 'cutoff': 500, 'q': 2.0}
            },
            
            'spectral_evolution': {
                'ca_type': 'HighLife',
                'grid_size': (45, 45),
                'density': 0.35,
                'generations': 25,
                'frequency_range': (300, 3000),
                'synthesis': 'FM',
                'fm_params': {'mod_freq': 5, 'mod_index': 3},
                'descriptor_control': {
                    'type': 'spectral_centroid',
                    'curve': [(1, 0.2), (3, 0.7), (5, 0.4)]
                }
            },
            
            'organic_texture': {
                'layers': [
                    {
                        'ca_type': 'Game of Life',
                        'frequency_range': (100, 300),
                        'waveform': 'sine',
                        'volume': 0.7
                    },
                    {
                        'ca_type': 'Coral',
                        'frequency_range': (800, 2000),
                        'waveform': 'triangle',
                        'volume': 0.5
                    },
                    {
                        'ca_type': 'Seeds',
                        'frequency_range': (2000, 4000),
                        'waveform': 'noise',
                        'volume': 0.3,
                        'filter': {'type': 'highpass', 'cutoff': 1500}
                    }
                ]
            }
        }
    
    def apply_preset(self, preset_name, target_layer=None):
        """Aplica preset a uma camada específica ou cria nova"""
        if preset_name not in self.presets:
            raise ValueError(f"Preset '{preset_name}' não encontrado")
        
        preset_config = self.presets[preset_name]
        
        if 'layers' in preset_config:
            # Preset multicamada
            return self.apply_multilayer_preset(preset_config)
        else:
            # Preset de camada única
            return self.apply_single_layer_preset(preset_config, target_layer)
    
    def save_current_as_preset(self, name, layers):
        """Salva configuração atual como novo preset"""
        config = self.extract_layer_config(layers)
        self.presets[name] = config
        self.save_presets_to_file()
    
    def get_available_presets(self):
        """Retorna lista de presets disponíveis com descrições"""
        descriptions = {
            'ambient_drone': 'Drone ambiental suave com evolução lenta',
            'rhythmic_pulse': 'Padrão rítmico orgânico com Seeds',
            'spectral_evolution': 'Evolução espectral controlada com FM',
            'organic_texture': 'Textura multicamada orgânica'
        }
        
        return [(name, descriptions.get(name, 'Sem descrição')) 
                for name in self.presets.keys()]
```

### Analisador de Complexidade Musical

```python
def analyze_musical_complexity(audio_sequence):
    """Analisa complexidade musical de sequência gerada"""
    
    # 1. Densidade Espectral
    spectral_density = []
    for frame in audio_sequence:
        fft = np.fft.fft(frame)
        power_spectrum = np.abs(fft)**2
        # Conta picos significativos
        peaks = find_spectral_peaks(power_spectrum)
        spectral_density.append(len(peaks))
    
    # 2. Variabilidade Temporal
    rms_values = [np.sqrt(np.mean(frame**2)) for frame in audio_sequence]
    temporal_variability = np.std(rms_values)
    
    # 3. Entropia Espectral
    spectral_entropy = []
    for frame in audio_sequence:
        spectrum = np.abs(np.fft.fft(frame))**2
        normalized_spectrum = spectrum / np.sum(spectrum)
        entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum + 1e-10))
        spectral_entropy.append(entropy)
    
    # 4. Índice de Harmonicidade
    harmonicity_index = calculate_harmonicity(audio_sequence)
    
    # 5. Complexidade Rítmica
    rhythmic_complexity = analyze_rhythmic_patterns(rms_values)
    
    complexity_metrics = {
        'spectral_density_mean': np.mean(spectral_density),
        'spectral_density_std': np.std(spectral_density),
        'temporal_variability': temporal_variability,
        'spectral_entropy_mean': np.mean(spectral_entropy),
        'harmonicity_index': harmonicity_index,
        'rhythmic_complexity': rhythmic_complexity,
        'overall_complexity': calculate_overall_complexity({
            'spectral': np.mean(spectral_density),
            'temporal': temporal_variability,
            'entropy': np.mean(spectral_entropy),
            'harmony': harmonicity_index,
            'rhythm': rhythmic_complexity
        })
    }
    
    return complexity_metrics

def generate_complexity_report(metrics):
    """Gera relatório textual de complexidade"""
    report = f"""
    === ANÁLISE DE COMPLEXIDADE MUSICAL ===
    
    Densidade Espectral: {metrics['spectral_density_mean']:.2f} ± {metrics['spectral_density_std']:.2f}
    Interpretação: {'Alta complexidade espectral' if metrics['spectral_density_mean'] > 50 else 'Complexidade espectral moderada'}
    
    Variabilidade Temporal: {metrics['temporal_variability']:.3f}
    Interpretação: {'Dinâmica expressiva' if metrics['temporal_variability'] > 0.1 else 'Dinâmica estável'}
    
    Entropia Espectral: {metrics['spectral_entropy_mean']:.2f}
    Interpretação: {'Textura rica e complexa' if metrics['spectral_entropy_mean'] > 10 else 'Textura mais simples'}
    
    Índice de Harmonicidade: {metrics['harmonicity_index']:.3f}
    Interpretação: {'Predominantemente harmônico' if metrics['harmonicity_index'] > 0.7 else 'Predominantemente inarmônico'}
    
    Complexidade Rítmica: {metrics['rhythmic_complexity']:.3f}
    Interpretação: {'Padrões rítmicos complexos' if metrics['rhythmic_complexity'] > 0.5 else 'Padrões rítmicos simples'}
    
    COMPLEXIDADE GERAL: {metrics['overall_complexity']:.3f}
    Classificação: {classify_complexity(metrics['overall_complexity'])}
    """
    
    return report

def classify_complexity(complexity_score):
    """Classifica nível de complexidade"""
    if complexity_score > 0.8:
        return "MUITO ALTA - Textura densa e complexa"
    elif complexity_score > 0.6:
        return "ALTA - Rica em detalhes espectrais"
    elif complexity_score > 0.4:
        return "MODERADA - Equilíbrio entre simplicidade e complexidade"
    elif complexity_score > 0.2:
        return "BAIXA - Textura simples e clara"
    else:
        return "MUITO BAIXA - Minimalista"
```

## Projetos de Pesquisa

### 13. Estudo Comparativo de CAs Musicais

**Metodologia**:
1. Gerar 100 amostras de cada tipo de CA
2. Analisar descritores estatísticos
3. Classificar por complexidade musical
4. Estudar correlações CA-Audio

**Implementação**:
```python
def comparative_ca_study():
    ca_types = ['Game of Life', 'Seeds', 'HighLife', 'Day and Night', 'Coral']
    results = {}
    
    for ca_type in ca_types:
        ca_samples = []
        audio_samples = []
        
        for i in range(100):
            # Parâmetros aleatórios dentro de ranges válidos
            config = generate_random_config(ca_type)
            
            # Gera CA e áudio
            ca_sequence = generate_ca_sequence(config)
            audio = ca_to_audio(ca_sequence, config)
            
            ca_samples.append(ca_sequence)
            audio_samples.append(audio)
        
        # Análise estatística
        results[ca_type] = {
            'ca_stats': analyze_ca_statistics(ca_samples),
            'audio_stats': analyze_audio_statistics(audio_samples),
            'complexity_metrics': [analyze_musical_complexity(audio) 
                                 for audio in audio_samples]
        }
    
    # Gera relatório comparativo
    generate_comparative_report(results)
    return results
```

### 14. Otimização por Algoritmos Genéticos

**Objetivo**: Encontrar configurações ótimas para características musicais específicas

```python
def optimize_for_musical_goal(target_characteristics):
    """
    Otimiza parâmetros CA para atingir características musicais específicas
    
    target_characteristics = {
        'spectral_centroid': 0.7,  # Som brilhante
        'rhythmic_regularity': 0.8,  # Ritmo regular
        'harmonic_content': 0.6,  # Moderadamente harmônico
        'temporal_evolution': 0.9  # Evolução pronunciada
    }
    """
    
    # Define espaço de parâmetros
    parameter_space = {
        'ca_type': ['Game of Life', 'Seeds', 'HighLife'],
        'density': (0.1, 0.8),
        'grid_size': (20, 80),
        'generations': (10, 50),
        'frequency_range': [(60, 500), (200, 2000), (1000, 4000)],
        'synthesis_type': ['basic', 'FM', 'AM'],
        'filter_cutoff': (200, 5000)
    }
    
    def fitness_function(individual):
        # Gera áudio com parâmetros do indivíduo
        audio = synthesize_with_parameters(individual)
        
        # Calcula características atuais
        current_chars = extract_musical_characteristics(audio)
        
        # Calcula distância dos objetivos
        fitness = 0
        for char, target_value in target_characteristics.items():
            current_value = current_chars.get(char, 0)
            error = abs(target_value - current_value)
            fitness += 1.0 / (1.0 + error)  # Fitness inverso do erro
        
        return fitness / len(target_characteristics)
    
    # Executa algoritmo genético
    ga = GeneticAlgorithm(
        population_size=50,
        parameter_space=parameter_space,
        fitness_function=fitness_function,
        generations=100
    )
    
    best_individual, best_fitness = ga.evolve()
    
    return best_individual, best_fitness
```

### 15. Sistema de Recomendação Musical

**Conceito**: Sistema que sugere configurações baseado em preferências do usuário

```python
class MusicalRecommendationSystem:
    def __init__(self):
        self.user_preferences = {}
        self.configuration_database = []
        self.rating_history = {}
    
    def rate_configuration(self, config, rating):
        """Usuário avalia configuração (1-5 estrelas)"""
        config_hash = self.hash_configuration(config)
        self.rating_history[config_hash] = rating
        
        # Atualiza preferências baseado na avaliação
        self.update_preferences(config, rating)
    
    def recommend_configurations(self, num_recommendations=5):
        """Recomenda configurações baseado em preferências"""
        scored_configs = []
        
        for config in self.configuration_database:
            score = self.calculate_preference_score(config)
            scored_configs.append((config, score))
        
        # Ordena por score e retorna top N
        scored_configs.sort(key=lambda x: x[1], reverse=True)
        return [config for config, score in scored_configs[:num_recommendations]]
    
    def calculate_preference_score(self, config):
        """Calcula score de preferência para uma configuração"""
        score = 0
        
        # Fatores baseados em preferências aprendidas
        for parameter, value in config.items():
            if parameter in self.user_preferences:
                preference_value = self.user_preferences[parameter]
                # Calcula similaridade (implementação específica depende do tipo de parâmetro)
                similarity = self.calculate_parameter_similarity(parameter, value, preference_value)
                score += similarity
        
        return score / len(config)
    
    def generate_personalized_variations(self, base_config):
        """Gera variações personalizadas de uma configuração base"""
        variations = []
        
        for _ in range(10):
            variation = base_config.copy()
            
            # Aplica mutações baseadas em preferências
            for parameter in variation:
                if random.random() < 0.3:  # 30% chance de mutação
                    variation[parameter] = self.mutate_parameter_intelligently(
                        parameter, variation[parameter]
                    )
            
            variations.append(variation)
        
        return variations
```

## Conclusão

Estes exemplos demonstram a versatilidade e potencial criativo do sintetizador de áudio baseado em autômatos celulares. Desde aplicações básicas até projetos de pesquisa avançados, o sistema oferece múltiplas abordagens para exploração sonora e composição algorítmica.

### Direções Futuras

1. **Machine Learning**: Integração de redes neurais para classificação automática de texturas sonoras
2. **Interatividade**: Desenvolvimento de interfaces gestuais e responsivas
3. **Escalabilidade**: Otimização para processamento em tempo real
4. **Colaboração**: Sistemas multi-usuário para composição coletiva
5. **Hibridização**: Combinação com outras técnicas de síntese (granular, espectral, etc.)

### Recursos para Aprofundamento

**Tutoriais em Vídeo**: [Links para demonstrações práticas]
**Samples de Áudio**: [Biblioteca de exemplos organizados por categoria]
**Patches Prontos**: [Configurações testadas para download]
**Comunidade**: [Fórum para compartilhamento de criações]

---

*Para mais exemplos e atualizações, visite o repositório oficial e participe da comunidade de usuários.* '