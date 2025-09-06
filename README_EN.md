# 🎵 Cellular Automaton Audio Synthesizer

A Python application for sound synthesis and musical composition based on Cellular Automata (CA) patterns. This project offers two distinct interfaces: a complete desktop application with Tkinter and a simplified web version with Streamlit.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This synthesizer converts evolutionary patterns from Cellular Automata into sound material, enabling:

- **Granular Synthesis**: Each living cell generates sound grains
- **Spatial Mapping**: X/Y coordinates determine frequencies
- **Temporal Evolution**: CA generations create temporal progressions
- **Multiple Layers**: Combination of different CA types
- **Descriptor Control**: Manipulation of spectral characteristics

### Supported Cellular Automata Types

- **Game of Life**: Conway's classic rules
- **HighLife**: Extension with replication
- **Seeds**: Simple explosive patterns
- **Day and Night**: Inverted rules
- **Anneal, Bacteria, Maze, Coral**: Specialized variations

## ✨ Features

### Desktop Version (Tkinter)
- Multi-tab interface with detailed controls
- Real-time synthesis with multiple layers
- Export audio, images, GIFs, and videos
- Advanced filter and effects control
- Interactive audio descriptor system
- Genetic algorithm for parameter optimization

### Web Version (Streamlit)
- Simplified and intuitive interface
- Real-time visualization
- Integrated audio player
- Basic spectral analysis
- Audio and animation export

## 🛠 Installation

### System Requirements
- Python 3.8 or higher
- Operating System: Windows, macOS, or Linux

### Main Dependencies

```bash
# Install dependencies
pip install -r requirements.txt
```

### Installation by Method

#### 1. Desktop Application

```bash
# Clone the repository
git clone https://github.com/your-username/ca-audio-synthesizer.git
cd ca-audio-synthesizer

# Install dependencies
pip install -r requirements.txt

# Run the application
python CA_SF_53_working_audio_descriptors_changers_analysis.py
```

#### 2. Web Application (Local)

```bash
# Run with Streamlit
streamlit run streamlit_ca_synthesizer.py
```

#### 3. Web Application (Streamlit Cloud)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://multi-layered-cellular-automaton-granular-synthesis-gui.streamlit.app)

**Direct access:** https://multi-layered-cellular-automaton-granular-synthesis-gui.streamlit.app

## 🎮 Usage

### Desktop Version

1. **Layer Configuration**:
   - Adjust the number of layers in the "Layer Settings" panel
   - Configure each layer individually

2. **CA Parameters**:
   - Choose cellular automaton type
   - Define grid dimensions (width/height)
   - Adjust initial density and number of generations

3. **Audio Parameters**:
   - Select waveform (Sine, Square, Sawtooth, etc.)
   - Configure frequency ranges (min/max)
   - Apply synthesis techniques (AM, FM, Ring Modulation)

4. **Global Controls**:
   - Generate new iterations
   - Play, pause, or stop audio
   - Export audio, image, or video files

### Web Version

1. **Sidebar Configuration**:
   - Select CA type and grid parameters
   - Adjust audio and filter parameters

2. **Generation**:
   - Click "Generate Audio" to create sound
   - Visualize CA evolution in the main panel

3. **Playback and Analysis**:
   - Use the integrated audio player
   - Analyze spectral characteristics
   - Download generated audio and animations

## 📚 Documentation

### Project Structure

```
ca-audio-synthesizer/
├── CA_SF_53_working_audio_descriptors_changers_analysis.py  # Desktop application
├── streamlit_ca_synthesizer.py                             # Web application
├── requirements.txt                                         # Dependencies
├── README.md                                               # Portuguese version
├── README_EN.md                                            # This file
├── docs/                                                   # Additional documentation
│   ├── user_guide.md                                      # User guide
│   ├── technical_details.md                               # Technical details
│   └── examples.md                                        # Usage examples
├── examples/                                              # Example files
│   ├── audio_samples/                                     # Audio samples
│   ├── ca_patterns/                                       # CA patterns
│   └── videos/                                            # Demo videos
└── assets/                                                # Visual resources
    ├── screenshots/                                       # Screenshots
    └── diagrams/                                          # Explanatory diagrams
```

### Synthesis Algorithm

1. **Initialization**: Random grid based on configured density
2. **Evolution**: Application of CA rules for N generations
3. **Mapping**: Conversion of coordinates to frequencies
4. **Synthesis**: Generation of sound grains for each living cell
5. **Processing**: Application of filters and effects
6. **Mixing**: Combination of multiple layers

### Audio Descriptors

- **Spectral Centroid**: Timbral brightness
- **Spectral Standard Deviation**: Spectral dispersion
- **Spectral Skewness**: Distribution asymmetry
- **Spectral Kurtosis**: Spectral concentration
- **RMS**: Root mean square amplitude
- **Spectral Flatness**: Tonal/noisy characteristics

## 🎵 Examples

### Basic Example (Game of Life)

```python
# Simple configuration for Game of Life
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

### Advanced Example (Multiple Layers)

```python
# Multi-layer configuration with different CAs
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

## 🔧 Advanced Parameters

### Synthesis Techniques
- **Amplitude Modulation (AM)**: Amplitude modulation
- **Frequency Modulation (FM)**: Frequency modulation  
- **Ring Modulation**: Ring modulation

### Audio Filters
- **Low Pass**: Low-pass filter
- **High Pass**: High-pass filter
- **Band Pass**: Band-pass filter

### Frequency Mappings
- **Standard**: Simple linear mapping
- **Complex**: Based on neighbor count
- **Rule-Based**: Based on CA rules
- **Interpolation**: Interpolation between neighbors
- **Harmonics**: Harmonic series

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

### Areas for Contribution

- New cellular automaton types
- Additional synthesis algorithms
- User interface improvements
- Performance optimizations
- Documentation and examples
- Automated testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by John Conway's work (Game of Life)
- Based on granular synthesis concepts
- Uses open-source libraries from the Python community

## 📞 Contact

- **Author**: [Your Name]
- **Email**: your.email@example.com
- **GitHub**: [@your-username](https://github.com/your-username)

---

*Created with ❤️ for the electronic music and creative programming community*