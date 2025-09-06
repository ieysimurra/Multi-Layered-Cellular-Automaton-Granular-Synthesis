import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import butter, sosfiltfilt
from scipy.io.wavfile import write
import librosa
import io
import tempfile
import os
import random
import imageio
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Constants
SAMPLE_RATE = 44100

# Streamlit page configuration
st.set_page_config(
    page_title="CA Audio Synthesizer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎵 Cellular Automaton Audio Synthesizer")
st.markdown("Generate music using cellular automata patterns")

class CAudioSynthesizer:
    def __init__(self):
        self.all_spaces = []
        self.generated_audio = None
        
    def generate_waveform(self, freq, duration, waveform_type, sample_rate=SAMPLE_RATE):
        """Generate different waveform types"""
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        if waveform_type == "Sine":
            return np.sin(2 * np.pi * freq * t)
        elif waveform_type == "Square":
            return scipy.signal.square(2 * np.pi * freq * t)
        elif waveform_type == "Sawtooth":
            return scipy.signal.sawtooth(2 * np.pi * freq * t)
        elif waveform_type == "Triangle":
            return scipy.signal.sawtooth(2 * np.pi * freq * t, 0.5)
        elif waveform_type == "White Noise":
            return np.random.uniform(-1, 1, int(sample_rate * duration))
        else:
            return np.sin(2 * np.pi * freq * t)
    
    def apply_filter(self, audio, filter_type, cutoff_freq, sample_rate=SAMPLE_RATE):
        """Apply audio filters"""
        try:
            if len(audio) < 20:
                return audio
                
            nyquist = sample_rate / 2.0
            normalized_cutoff = min(0.99, cutoff_freq / nyquist)
            
            if normalized_cutoff <= 0:
                return audio
                
            sos = butter(N=2, Wn=normalized_cutoff, btype=filter_type.lower(), output='sos')
            
            if audio.ndim == 2:
                filtered_audio = np.zeros_like(audio)
                for channel in range(audio.shape[1]):
                    filtered_audio[:, channel] = sosfiltfilt(sos, audio[:, channel])
                return filtered_audio
            else:
                return sosfiltfilt(sos, audio)
        except Exception as e:
            st.error(f"Filter error: {e}")
            return audio
    
    def calculate_spectral_centroid(self, audio):
        """Calculate spectral centroid for audio descriptor control"""
        try:
            if audio.ndim == 2:
                audio_mono = np.mean(audio, axis=1)
            else:
                audio_mono = audio
                
            if len(audio_mono) < 1024:
                return 0.5
                
            centroid = librosa.feature.spectral_centroid(y=audio_mono, sr=SAMPLE_RATE)[0].mean()
            normalized = (centroid - 60) / (6000 - 60)
            return np.clip(normalized, 0, 1)
        except:
            return 0.5
    
    def game_of_life_step(self, grid):
        """Conway's Game of Life rules"""
        new_grid = np.zeros_like(grid)
        rows, cols = grid.shape
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                neighbors = np.sum(grid[i-1:i+2, j-1:j+2]) - grid[i, j]
                if grid[i, j] == 1 and neighbors in [2, 3]:
                    new_grid[i, j] = 1
                elif grid[i, j] == 0 and neighbors == 3:
                    new_grid[i, j] = 1
        return new_grid
    
    def seeds_step(self, grid):
        """Seeds CA rules"""
        new_grid = np.zeros_like(grid)
        rows, cols = grid.shape
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                neighbors = np.sum(grid[i-1:i+2, j-1:j+2]) - grid[i, j]
                if grid[i, j] == 0 and neighbors == 2:
                    new_grid[i, j] = 1
        return new_grid
    
    def highlife_step(self, grid):
        """HighLife CA rules"""
        new_grid = np.zeros_like(grid)
        rows, cols = grid.shape
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                neighbors = np.sum(grid[i-1:i+2, j-1:j+2]) - grid[i, j]
                if grid[i, j] == 1 and neighbors in [2, 3]:
                    new_grid[i, j] = 1
                elif grid[i, j] == 0 and neighbors in [3, 6]:
                    new_grid[i, j] = 1
        return new_grid
    
    def generate_ca_sequence(self, ca_type, width, height, generations, density):
        """Generate cellular automaton sequence"""
        # Initialize random grid
        initial_grid = np.random.choice([0, 1], size=(width, height), p=[1-density, density])
        
        # Store all generations
        sequence = [initial_grid.copy()]
        current_grid = initial_grid.copy()
        
        # CA rules mapping
        ca_rules = {
            "Game of Life": self.game_of_life_step,
            "Seeds": self.seeds_step,
            "HighLife": self.highlife_step
        }
        
        step_function = ca_rules.get(ca_type, self.game_of_life_step)
        
        for _ in range(generations - 1):
            current_grid = step_function(current_grid)
            sequence.append(current_grid.copy())
        
        return np.array(sequence)
    
    def ca_to_audio(self, ca_sequence, params):
        """Convert CA sequence to audio"""
        generations, width, height = ca_sequence.shape
        total_duration = params['instance_duration']
        frame_duration = total_duration / generations
        
        all_audio = []
        
        for gen_idx, generation in enumerate(ca_sequence):
            frame_audio = np.zeros(int(frame_duration * SAMPLE_RATE))
            
            for x in range(width):
                for y in range(height):
                    if generation[x, y] == 1:
                        # Map position to frequency
                        freq_x = np.interp(x, [0, width-1], [params['min_freq'], params['max_freq']])
                        freq_y = np.interp(y, [0, height-1], [params['min_freq'], params['max_freq']])
                        freq = np.sqrt(freq_x * freq_y)
                        
                        # Generate grain
                        grain = self.generate_waveform(freq, frame_duration, params['waveform'])
                        
                        # Apply amplitude based on position
                        amp = np.interp(x + y, [0, width + height - 2], [0.1, 0.8])
                        grain *= amp
                        
                        # Add to frame audio
                        frame_audio += grain[:len(frame_audio)]
            
            # Normalize frame
            if np.max(np.abs(frame_audio)) > 0:
                frame_audio /= np.max(np.abs(frame_audio))
            
            all_audio.append(frame_audio)
        
        # Concatenate all frames
        final_audio = np.concatenate(all_audio)
        
        # Apply filter if specified
        if params.get('filter_enabled', False):
            final_audio = self.apply_filter(
                final_audio, 
                params['filter_type'], 
                params['filter_cutoff']
            )
        
        # Convert to stereo
        if final_audio.ndim == 1:
            stereo_audio = np.column_stack([final_audio, final_audio])
        else:
            stereo_audio = final_audio
            
        # Normalize final audio
        if np.max(np.abs(stereo_audio)) > 0:
            stereo_audio /= np.max(np.abs(stereo_audio))
        
        return stereo_audio

# Initialize synthesizer
@st.cache_resource
def get_synthesizer():
    return CAudioSynthesizer()

synth = get_synthesizer()

# Sidebar controls
st.sidebar.header("🎛️ Controls")

# CA Parameters
st.sidebar.subheader("Cellular Automaton")
ca_type = st.sidebar.selectbox(
    "CA Type",
    ["Game of Life", "Seeds", "HighLife"]
)

col1, col2 = st.sidebar.columns(2)
with col1:
    width = st.number_input("Width", min_value=10, max_value=100, value=30)
with col2:
    height = st.number_input("Height", min_value=10, max_value=100, value=30)

generations = st.sidebar.slider("Generations", min_value=5, max_value=50, value=20)
density = st.sidebar.slider("Initial Density", min_value=0.1, max_value=0.9, value=0.3)

# Audio Parameters
st.sidebar.subheader("🎵 Audio")
duration = st.sidebar.slider("Duration (seconds)", min_value=0.5, max_value=10.0, value=2.0)
waveform = st.sidebar.selectbox(
    "Waveform",
    ["Sine", "Square", "Sawtooth", "Triangle", "White Noise"]
)

col3, col4 = st.sidebar.columns(2)
with col3:
    min_freq = st.number_input("Min Freq (Hz)", min_value=50, max_value=1000, value=200)
with col4:
    max_freq = st.number_input("Max Freq (Hz)", min_value=200, max_value=4000, value=1000)

# Filter controls
st.sidebar.subheader("🎚️ Filter")
filter_enabled = st.sidebar.checkbox("Enable Filter")
if filter_enabled:
    filter_type = st.sidebar.selectbox("Filter Type", ["lowpass", "highpass", "bandpass"])
    filter_cutoff = st.sidebar.slider("Cutoff Freq (Hz)", min_value=100, max_value=5000, value=1000)

# Generate button
if st.sidebar.button("🎼 Generate Audio", type="primary"):
    with st.spinner("Generating cellular automaton and audio..."):
        # Create parameter dictionary
        params = {
            'instance_duration': duration,
            'min_freq': min_freq,
            'max_freq': max_freq,
            'waveform': waveform,
            'filter_enabled': filter_enabled
        }
        
        if filter_enabled:
            params['filter_type'] = filter_type
            params['filter_cutoff'] = filter_cutoff
        
        # Generate CA sequence
        ca_sequence = synth.generate_ca_sequence(
            ca_type, width, height, generations, density
        )
        synth.all_spaces = [ca_sequence]
        
        # Generate audio
        audio = synth.ca_to_audio(ca_sequence, params)
        synth.generated_audio = audio
        
        st.success("Audio generated successfully!")

# Main content area
col_left, col_right = st.columns([2, 1])

with col_left:
    st.header("📊 Visualization")
    
    if hasattr(synth, 'all_spaces') and synth.all_spaces:
        ca_data = synth.all_spaces[0]
        
        # Create animation slider
        if len(ca_data) > 1:
            frame_idx = st.slider(
                "Generation", 
                min_value=0, 
                max_value=len(ca_data)-1, 
                value=0
            )
            
            # Display current frame
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(ca_data[frame_idx], cmap='binary', aspect='equal')
            ax.set_title(f'{ca_type} - Generation {frame_idx + 1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im)
            st.pyplot(fig)
            plt.close()
            
            # Animation controls
            if st.button("🎬 Create Animation"):
                with st.spinner("Creating animation..."):
                    # Create GIF
                    frames = []
                    for i, generation in enumerate(ca_data):
                        fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
                        ax.imshow(generation, cmap='binary', aspect='equal')
                        ax.set_title(f'{ca_type} - Generation {i + 1}')
                        ax.axis('off')
                        
                        # Save frame to bytes
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight')
                        buf.seek(0)
                        frames.append(imageio.imread(buf))
                        plt.close(fig)
                        buf.close()
                    
                    # Create GIF
                    gif_buffer = io.BytesIO()
                    imageio.mimsave(gif_buffer, frames, format='GIF', duration=0.2)
                    gif_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download Animation",
                        data=gif_buffer.getvalue(),
                        file_name=f"ca_animation_{ca_type.lower().replace(' ', '_')}.gif",
                        mime="image/gif"
                    )
    else:
        st.info("👆 Generate audio to see visualization")

with col_right:
    st.header("🎧 Audio Player")
    
    if synth.generated_audio is not None:
        # Audio player
        audio_data = synth.generated_audio
        
        # Create audio file in memory
        audio_buffer = io.BytesIO()
        
        # Convert to int16 for WAV format
        audio_int16 = (audio_data * 32767).astype(np.int16)
        write(audio_buffer, SAMPLE_RATE, audio_int16)
        audio_buffer.seek(0)
        
        st.audio(audio_buffer.getvalue(), format='audio/wav')
        
        # Download button
        st.download_button(
            label="📥 Download Audio",
            data=audio_buffer.getvalue(),
            file_name=f"ca_audio_{ca_type.lower().replace(' ', '_')}.wav",
            mime="audio/wav"
        )
        
        # Audio analysis
        st.subheader("📈 Audio Analysis")
        
        # Calculate spectral centroid
        centroid = synth.calculate_spectral_centroid(audio_data)
        st.metric("Spectral Centroid", f"{centroid:.3f}")
        
        # Show waveform
        if audio_data.ndim == 2:
            waveform_data = audio_data[:, 0]  # Left channel
        else:
            waveform_data = audio_data
            
        # Sample for display (to avoid performance issues)
        display_samples = min(len(waveform_data), 44100)  # 1 second max
        sample_indices = np.linspace(0, len(waveform_data)-1, display_samples, dtype=int)
        display_waveform = waveform_data[sample_indices]
        time_axis = np.linspace(0, len(waveform_data)/SAMPLE_RATE, display_samples)
        
        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(
            x=time_axis,
            y=display_waveform,
            mode='lines',
            name='Waveform',
            line=dict(width=1)
        ))
        fig_wave.update_layout(
            title="Audio Waveform",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=300
        )
        st.plotly_chart(fig_wave, use_container_width=True)
        
    else:
        st.info("👆 Generate audio to see player and analysis")

# Additional info
with st.expander("ℹ️ About"):
    st.markdown("""
    **Cellular Automaton Audio Synthesizer**
    
    This application generates music by converting cellular automaton patterns into audio:
    
    1. **Cellular Automaton**: Choose from different CA rules that create evolving patterns
    2. **Audio Mapping**: Each live cell generates a tone based on its position
    3. **Frequency Mapping**: X/Y coordinates map to frequency ranges
    4. **Temporal Evolution**: CA generations become audio frames over time
    
    **CA Types:**
    - **Game of Life**: Classic Conway's rules (birth: 3, survival: 2-3)
    - **Seeds**: Simple explosive patterns (birth: 2, survival: none)
    - **HighLife**: Extended life with replication (birth: 3,6, survival: 2-3)
    
    **Tips:**
    - Higher density creates more complex sounds
    - Adjust frequency range for different tonal qualities
    - Use filters to shape the timbre
    - Try different waveforms for various textures
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Cellular Automaton Audio Synthesis")
