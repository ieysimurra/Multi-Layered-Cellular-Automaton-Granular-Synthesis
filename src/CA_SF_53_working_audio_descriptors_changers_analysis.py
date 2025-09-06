import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import scipy
import scipy.signal
from scipy.signal import butter, sosfiltfilt, sosfreqz
from scipy.io.wavfile import write
from scipy.interpolate import interp1d
import pyroomacoustics as pra
import librosa
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import Scale, HORIZONTAL, Label
from tkinter import BooleanVar
from threading import Thread
import imageio
import os
import random
from sklearn.decomposition import PCA

# Constants for audio generation
SAMPLE_RATE = 44100
GRAIN_DURATION = 1.0  # 20ms in seconds
width, height = 50, 50  # Grid dimensions
instance_num = 0
# Define the parameter space
waveform_space = ["Sine", "Triangle", "Sawtooth", "Square", "Reverse Sawtooth", "Pulse", "White Noise"]
freqMapping_space = ["Standard", "Complex Frequency Mapping", "CA Rule-Based Frequency", "Frequency Interpolation", "Harmonics and Overtones"]
synTech_space = ["None", "Ring Modulation", "Amplitude Modulation", "Frequency Modulation"]
amFreq_space = np.linspace(20, 2000, 10)  # From 20 Hz to 2000 Hz, 10 steps
rmFreq_space = np.linspace(500, 5000, 10)  # From 500 Hz to 5000 Hz, 10 steps
fmFreq_space = np.linspace(20, 2000, 10)  # From 20 Hz to 2000 Hz, 10 steps
fmIndex_space = np.linspace(1, 20, 10)  # From 1 Hz to 20 Hz, 10 steps
filter_space = ["LowPass", "HighPass", "BandPass"]
cutFreq_space = np.linspace(500, 5000, 10)  # From 500 Hz to 5000 Hz, 10 steps
num_steps = int((10.0 - 0.1) / 0.01) + 1  # Convert to int for safety
qFactor_space = np.linspace(0.1, 10.0, num_steps)  # From 0.1 to 10.0 Hz, 0.01 steps
mindex_space = np.linspace(0.1, 2.0, 10)  # From 0.1 to 2.0, 10 steps

class GlobalControlsTab(ttk.Frame):
    def __init__(self, master, main_gui):
        super().__init__(master)
        self.main_gui = main_gui
        # Add a variable to store the synthesis mode (manual or GA)
        self.synthesis_mode_var = tk.StringVar(value="Manual")  # Default to manual mode
        self.init_ui()
        self.layer_audios = []
        self.mixed_audio = None  # Initialize mixed_audio here

    def init_ui(self):

        # Reverb effect variables
        self.reverb_enabled_var = tk.BooleanVar(value=False)
        self.reverb_wet_var = tk.DoubleVar(value=0.5)  # Default 50% wet
        self.reverb_damping_var = tk.DoubleVar(value=0.5)  # Default 50% damping
        self.reverb_room_size_var = tk.DoubleVar(value=50)  # Default room size, normalized 1-100

        # Reverb effect controls
        ttk.Checkbutton(self, text="Enable Reverb", variable=self.reverb_enabled_var).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ttk.Label(self, text="Wet Level:").grid(row=0, column=1, padx=10, pady=5, sticky="w")
        tk.Scale(self, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, variable=self.reverb_wet_var).grid(row=0, column=2, padx=10, pady=5, sticky="ew")
        ttk.Label(self, text="Damping:").grid(row=0, column=3, padx=10, pady=5, sticky="w")
        tk.Scale(self, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, variable=self.reverb_damping_var).grid(row=0, column=4, padx=10, pady=5, sticky="ew")
        ttk.Label(self, text="Room Size:").grid(row=0, column=5, padx=10, pady=5, sticky="w")
        tk.Scale(self, from_=1, to=100, resolution=0.1, orient=tk.HORIZONTAL, variable=self.reverb_room_size_var).grid(row=0, column=6, padx=10, pady=5, sticky="ew")        

        self.filename_var = tk.StringVar(value="output")

        # Synthesis Mode Selection
        ttk.Label(self, text="Synthesis Mode:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        synthesis_mode_dropdown = ttk.Combobox(self, textvariable=self.synthesis_mode_var, 
                                            values=("Manual", "GA (Collect Y-Values)"), state="readonly")
        synthesis_mode_dropdown.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # Add global controls like Play, Stop, Export, etc.
        ttk.Button(self, text="Generate New Iterations", command=self.on_generate_new_iterations).grid(row=2, column=0, padx=10, pady=10)
        # Play, Pause, Stop Buttons
        ttk.Button(self, text="Play", command=self.play_audio).grid(row=3, column=0, padx=10, pady=10)
        ttk.Button(self, text="Pause", command=self.pause_audio).grid(row=3, column=1, padx=10, pady=10)
        ttk.Button(self, text="Stop", command=self.stop_audio).grid(row=3, column=2, padx=10, pady=10)

        # Filename Entry and Export Buttons
        ttk.Button(self, text="Export Audio", command=self.export_audio).grid(row=4, column=0, padx=10, pady=10)
        ttk.Button(self, text="Export Images", command=self.export_images).grid(row=4, column=1, padx=10, pady=10)
        ttk.Button(self, text="Export GIF", command=self.export_gifs).grid(row=4, column=2, padx=10, pady=10)
        ttk.Button(self, text="Export Video", command=self.export_videos).grid(row=4, column=3, padx=10, pady=10)
        ttk.Label(self, text="Filename: ").grid(row=5, column=0, padx=10, pady=5)
        ttk.Entry(self, textvariable=self.filename_var).grid(row=5, column=1, padx=10, pady=10)

        # Status Message Label
        self.status_label = ttk.Label(self, text="")
        self.status_label.grid(row=7, column=0, columnspan=4, padx=10, pady=10)  # Adjust grid parameters as needed

        # Add more buttons and controls as needed

    def initialize_ga(self):
        self.population = self.initialize_population(self.pop_size, self.param_space)
        self.current_generation = 0

    def on_generate_new_iterations(self):

        synthesis_mode = self.synthesis_mode_var.get()
        if synthesis_mode == "Manual":
            self.process_manual_synthesis()
        elif synthesis_mode == "GA (Collect Y-Values)":
            self.process_ga_synthesis()

    def process_manual_synthesis(self):
        all_layers_audio = []
        self.layer_audios = []  # Store each layer's audio
        # Existing code for manual synthesis goes here
        # Iterate through each CALayerTab and generate audio
        for tab in self.main_gui.notebook.tabs()[2:]:  # Skip fixed tabs
            layer_tab = self.main_gui.notebook.nametowidget(tab)
            layer_audio = layer_tab.generate_audio_for_layer()
            all_layers_audio.append(layer_audio)
            self.layer_audios.append(layer_audio)  # Store the layer audio    
            
        self.mixed_audio = self.mix_audio(all_layers_audio)

        # At the end of the generation process
        self.status_label["text"] = "New iterations generated successfully!"

    def process_ga_synthesis(self):
        # Assume each layer has its own set of targets and parameters
        # For demonstration, let's define some placeholders for GA parameters
        generations = 50  # Number of GA generations
        pop_size = 20  # Population size
        # Parameter space example: {parameter_name: [min_value, max_value]}
        param_space = {
            'modulation_frequency': [20, 2000],
            'modulation_index': [0.1, 10],
            # Add more synthesis parameters as required
        }
        # Placeholder for target centroids, should be dynamically obtained
        target_centroids = self.get_dynamic_target_centroids()

        # Execute GA for each target centroid
        for target_centroid in target_centroids:
            # Adjust run_genetic_algorithm to accept and work with the single target_centroid
            best_parameters, best_fitness = self.run_genetic_algorithm(
                generations, pop_size, param_space, target_centroid)
            
            # Use the best_parameters from GA to synthesize sound
            # This step depends on your synthesis method. As a placeholder:
            synthesized_sound = self.synthesize_sound(best_parameters)
            # Do something with the synthesized sound, e.g., mix it into the layer's audio stream
            self.mix_into_layer_audio(synthesized_sound)

        # After processing all centroids, update the GUI or perform additional steps as needed

        print("Not implemented yet")

    def apply_reverb(self, audio, sample_rate=44100):
        if not self.reverb_enabled_var.get():
            return audio  # Return unchanged audio if reverb is not enabled

        # Convert stereo audio to mono if necessary
        if audio.ndim == 2 and audio.shape[1] == 2:
            audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio  # Already mono or single channel

        # Calculate room dimensions based on room size factor
        room_size_factor = self.reverb_room_size_var.get() / 100
        room_dimensions = [5 + 15 * room_size_factor, 4 + 11 * room_size_factor, 2.5 + 7.5 * room_size_factor]

        # Example positions based on room dimensions
        mic_position = [room_dimensions[0] / 2, room_dimensions[1] / 2, 1.2]  # Centered at half the height
        source_position = [room_dimensions[0] / 4, room_dimensions[1] / 4, 1.2]  # Quarter way into the room

        # Adjust absorption based on damping control
        absorption = 0.2 + 0.5 * (1 - self.reverb_damping_var.get())

        # Create a shoebox room with the specified dimensions and absorption
        shoebox = pra.ShoeBox(room_dimensions, fs=sample_rate, absorption=absorption, max_order=15)

        # Add the source and microphone to the room
        shoebox.add_source(position=source_position, signal=audio_mono)
        shoebox.add_microphone_array(pra.MicrophoneArray(np.array([mic_position]).T, fs=sample_rate))

        # Run the simulation
        shoebox.simulate()

        # Get the simulated signal with reverb
        reverberated_signal = shoebox.mic_array.signals[0, :]

        # Mix the dry and wet signals based on the wet level control
        wet = self.reverb_wet_var.get()
        
        # Option 1: Padding the shorter array
        if len(audio_mono) < len(reverberated_signal):
            padded_audio_mono = np.pad(audio_mono, (0, len(reverberated_signal) - len(audio_mono)), mode='constant')
            mixed_audio = padded_audio_mono * (1 - self.reverb_wet_var.get()) + reverberated_signal * self.reverb_wet_var.get()
        else:
            padded_reverberated_signal = np.pad(reverberated_signal, (0, len(audio_mono) - len(reverberated_signal)), mode='constant')
            mixed_audio = audio_mono * (1 - self.reverb_wet_var.get()) + padded_reverberated_signal * self.reverb_wet_var.get()

        # Normalize the final mixed audio
        max_amplitude = np.max(np.abs(mixed_audio))
        if max_amplitude > 0:
            mixed_audio /= max_amplitude

        return mixed_audio

    def mix_audio(self, audio_layers):
        # Ensure each audio layer has the same length for proper mixing
        max_length = max(len(layer) for layer in audio_layers)
        padded_layers = [np.pad(layer, ((0, max_length - len(layer)), (0, 0)), mode='constant') for layer in audio_layers]
        
        # Mix the audio arrays from all layers
        mixed_audio = np.sum(padded_layers, axis=0) / len(padded_layers)

        # Normalize the mixed audio
        max_amplitude = np.max(np.abs(mixed_audio))
        if max_amplitude > 0:
            mixed_audio = mixed_audio / max_amplitude

        return mixed_audio

    def play_audio(self):
        if self.mixed_audio is not None:
            # Check if reverb is enabled and apply reverb effect
            if self.reverb_enabled_var.get():
                mixed_audio_with_reverb = self.apply_reverb(self.mixed_audio, SAMPLE_RATE)
            else:
                mixed_audio_with_reverb = self.mixed_audio

            sd.stop()  # Stop any currently playing audio
            # Play the audio with or without reverb as determined above
            self.play_thread = Thread(target=lambda: sd.play(mixed_audio_with_reverb, SAMPLE_RATE))
            self.play_thread.start()

    def pause_audio(self):
        sd.stop()

    def stop_audio(self):
        sd.stop()
        self.all_audio = np.array([])

    def export_audio(self):
        try:
            filename = self.filename_var.get()
            if not filename:
                raise ValueError("Filename is empty")

                # Apply reverb to the mixed_audio if enabled
            if self.reverb_enabled_var.get():
                self.mixed_audio = self.apply_reverb(self.mixed_audio, SAMPLE_RATE)

            # Export mixed audio
            if self.mixed_audio is None or self.mixed_audio.size == 0:
                self.status_label["text"] = "No mixed audio to export."
            else:
                mixed_audio_to_save = (self.mixed_audio * 32767).astype(np.int16)
                write(filename + "_mixed.wav", SAMPLE_RATE, mixed_audio_to_save)

            # Export each layer's audio
            for i, layer_audio in enumerate(self.layer_audios, start=1):
                if layer_audio.size > 0:
                    layer_audio_to_save = (layer_audio * 32767).astype(np.int16)
                    write(f"{filename}_l{i}.wav", SAMPLE_RATE, layer_audio_to_save)

            self.status_label["text"] = "All audios exported successfully."
        except Exception as e:
            self.status_label["text"] = f"Error in exporting audio: {e}"

    def export_images(self):
        try:
            filename = self.filename_var.get()
            if not filename:
                raise ValueError("Filename is empty")

            for i, tab in enumerate(self.main_gui.notebook.tabs()[2:], start=1):
                layer_tab = self.main_gui.notebook.nametowidget(tab)

                # Generate a filename prefix for each layer
                filename_prefix = f'{filename}_layer_{i}'

                # Call export_images on each CALayerTab with the correct filename prefix
                layer_tab.export_images(filename_prefix)

            self.status_label["text"] = "Images exported successfully."
        except Exception as e:
            self.status_label["text"] = f"Error in exporting images: {e}"

    def export_gifs(self):
        try:
            filename = self.filename_var.get()
            if not filename:
                raise ValueError("Filename is empty")

            for i, tab in enumerate(self.main_gui.notebook.tabs()[2:], start=1):
                layer_tab = self.main_gui.notebook.nametowidget(tab)

                # Generate a filename prefix for each layer
                filename_prefix = f'{filename}_layer_{i}'

                # Call export_images on each CALayerTab with the correct filename prefix
                layer_tab.export_gifs(filename_prefix)

            self.status_label["text"] = "Gifs exported successfully."
        except Exception as e:
            self.status_label["text"] = f"Error in exporting gifs: {e}"      

    def export_videos(self):
        try:
            filename = self.filename_var.get()
            if not filename:
                raise ValueError("Filename is empty")

            for i, tab in enumerate(self.main_gui.notebook.tabs()[2:], start=1):
                layer_tab = self.main_gui.notebook.nametowidget(tab)
                
                if not hasattr(layer_tab, 'all_spaces') or not layer_tab.all_spaces:
                    print(f"Layer {i}: No CA data to export")
                    continue
                    
                grid_width = layer_tab.width_var.get()
                grid_height = layer_tab.height_var.get()
                aspect_ratio = grid_width / grid_height

                for instance_num, space in enumerate(layer_tab.all_spaces):
                    video_filename = f'{filename}_layer_{i}_instance_{instance_num+1}.mp4'
                    fps = 30
                    
                    print(f"Exporting video: {video_filename}")
                    
                    with imageio.get_writer(video_filename, fps=fps, format='mp4', 
                                        codec='libx264', quality=10) as writer:
                        for frame_idx, generation in enumerate(space):
                            try:
                                # MÉTODO UNIVERSAL - Salvando como imagem temporária
                                import tempfile
                                import os
                                
                                # Criar figura
                                fig, ax = plt.subplots(figsize=(8 * aspect_ratio, 8), dpi=100)
                                ax.imshow(generation, cmap='binary', aspect='auto')
                                ax.set_title(f'2D Cellular Automaton (Layer {i}, Instance {instance_num + 1})')
                                ax.axis('off')
                                
                                # Salvar como arquivo temporário
                                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                                    fig.savefig(tmp_file.name, format='png', bbox_inches='tight', 
                                            facecolor='white', edgecolor='none', dpi=100)
                                    plt.close(fig)
                                    
                                    # Ler a imagem como array numpy
                                    temp_image = imageio.imread(tmp_file.name)
                                    
                                    # Remover arquivo temporário
                                    os.unlink(tmp_file.name)
                                    
                                    # Adicionar frame ao vídeo
                                    writer.append_data(temp_image)
                                
                            except Exception as e:
                                print(f"Error processing frame {frame_idx}: {e}")
                                plt.close('all')
                                continue

            self.status_label["text"] = "Video files exported successfully."
            print("All videos exported successfully!")

        except Exception as e:
            print(f"Error exporting video: {e}")
            self.status_label["text"] = f"Error exporting video: {e}"
            plt.close('all')

    def export_videos_alternative(self):
        """
        Método alternativo caso tostring_rgb não funcione
        """
        try:
            filename = self.filename_var.get()
            if not filename:
                raise ValueError("Filename is empty")

            for i, tab in enumerate(self.main_gui.notebook.tabs()[2:], start=1):
                layer_tab = self.main_gui.notebook.nametowidget(tab)
                
                if not hasattr(layer_tab, 'all_spaces') or not layer_tab.all_spaces:
                    continue
                    
                grid_width = layer_tab.width_var.get()
                grid_height = layer_tab.height_var.get()
                aspect_ratio = grid_width / grid_height

                for instance_num, space in enumerate(layer_tab.all_spaces):
                    video_filename = f'{filename}_layer_{i}_instance_{instance_num+1}.mp4'
                    fps = 30
                    
                    with imageio.get_writer(video_filename, fps=fps, format='mp4', 
                                        codec='libx264', quality=10) as writer:
                        for generation in space:
                            try:
                                fig, ax = plt.subplots(figsize=(8 * aspect_ratio, 8), dpi=100)
                                ax.imshow(generation, cmap='binary', aspect='auto')
                                ax.set_title(f'2D Cellular Automaton (Layer {i}, Instance {instance_num + 1})')
                                ax.axis('off')

                                # MÉTODO ALTERNATIVO usando buffer_rgba
                                fig.canvas.draw()
                                
                                # Tentar tostring_rgb primeiro
                                try:
                                    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                                    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                                except AttributeError:
                                    # Fallback para buffer_rgba se tostring_rgb não existir
                                    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                                    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                                    buf = buf[:, :, :3]  # Remover canal alpha para RGB
                                
                                writer.append_data(buf)
                                plt.close(fig)
                                
                            except Exception as e:
                                print(f"Error processing frame: {e}")
                                plt.close('all')
                                continue

            self.status_label["text"] = "Video files exported successfully."

        except Exception as e:
            print(f"Error exporting video: {e}")
            self.status_label["text"] = f"Error exporting video: {e}"            

class CALayerTab(ttk.Frame):
    
    def __init__(self, master, layer_num, update_synthesis_callback):
        super().__init__(master)
        self.layer_num = layer_num
        self.update_synthesis_callback = update_synthesis_callback

        # New variables for min and max frequency
        self.min_freq_var = tk.DoubleVar(value=60)  # Default minimum frequency
        self.max_freq_var = tk.DoubleVar(value=3600)  # Default maximum frequency

        # Define synthesis_technique_var before calling init_ui
        self.synthesis_technique_var = tk.StringVar(value="None")

        self.update_synthesis_callback = update_synthesis_callback  # Callback to update synthesis parameters in the main synthesis engine
    
        self.drawn_points = []  # To store the points
        self.mixed_audio = None  # To store the mixed audio data
        self.init_ui()

    def on_done_button_clicked(self):
        # Extract y-values from the drawn points and print them
        y_values = [y for _, y in self.drawn_points]
        print(f"Layer {self.layer_num} y-values:", y_values)

    def reset_layer_settings(self):
        # Reset settings to their default values
        self.grain_duration_var.set(0.02)
        self.num_generations_var.set(10)
        self.num_instances_var.set(5)
        self.num_instances_var.trace("w", self.clear_points_and_update_canvas)
        self.width_var.set(50)
        self.height_var.set(50)
        self.random_duration_var.set(False)
        self.min_random_duration_var.set(0.1)
        self.max_random_duration_var.set(1.0)
        self.fixed_time_span_var.set(0.5)
        self.random_time_span_var.set(False)
        self.min_random_time_span_var.set(0.1)
        self.max_random_time_span_var.set(1.0)        

        # Update the widgets to reflect the reset values
        # For example:
        # self.some_slider.set(self.some_variable.get())

    def init_ui(self):

        # Variables
        self.grain_duration_var = tk.DoubleVar(value=0.02)
        self.num_generations_var = tk.IntVar(value=10)
        self.num_instances_var = tk.IntVar(value=5)

        # Creating two new Tkinter variables, one for width and another for height.
        self.width_var = tk.IntVar(value=50)  # Initial value
        self.height_var = tk.IntVar(value=50)  # Initial value

        # New variable to hold the state of the checkbox
        self.random_duration_var = BooleanVar(value=False)
        # New: Create sliders for min and max random duration values
        self.min_random_duration_var = tk.DoubleVar(value=0.1)
        self.max_random_duration_var = tk.DoubleVar(value=1.0)

        self.fixed_time_span_var = tk.DoubleVar(value=0.5)
        self.random_time_span_var = BooleanVar(value=False)
        self.min_random_time_span_var = tk.DoubleVar(value=0.1)
        self.max_random_time_span_var = tk.DoubleVar(value=1.0)

        # Initialization including the Canvas creation...
        self.drawn_points = []  # Initialize an empty list to store points

        # Status Message Label
        self.status_label = ttk.Label(self, text="")
        self.status_label.grid(row=16, column=4, padx=10, pady=10)

        # Audio Stream
        self.all_audio = np.array([])

        self.all_spaces = []        
        # CA Type Dropdown
        ttk.Label(self, text="CA Type: ").grid(row=0, column=0, padx=10, pady=5)
        self.ca_type_var = tk.StringVar(value="Game of Life")
        ca_type_dropdown = ttk.Combobox(self, textvariable=self.ca_type_var, values=(
            "Game of Life", "HighLife", "Seeds", "Day and Night", "Anneal", 
            "Bacteria", "Maze", "Coral", "Exploding Rules"))
        ca_type_dropdown.grid(row=0, column=1, padx=10, pady=5)

        # Initial Grid Setting Dropdown
        self.init_grid_var = tk.StringVar(value="Random")
        self.init_grid_percent_var = tk.DoubleVar(value=0.75)
        ttk.Label(self, text="initial Grid: ").grid(row=2, column=0, padx=10, pady=5)
        init_grid_dropdown = ttk.Combobox(self, textvariable=self.init_grid_var, values=("Random"))
        init_grid_dropdown.grid(row=2, column=1, padx=10, pady=5)

        # Entry for percentage of 1s in initial grid
        self.init_grid_percent_entry = ttk.Entry(self, textvariable=self.init_grid_percent_var)
        self.init_grid_percent_entry.grid(row=2, column=2, padx=10, pady=5)
        ttk.Label(self, text="% of 1s: ").grid(row=2, column=3, padx=10, pady=5)            

        # Sliders for width and height
        self.width_var = tk.IntVar(value=50)        
        ttk.Label(self, text="Grid Width: ").grid(row=3, column=0, padx=10, pady=5)        
        tk.Scale(self, variable=self.width_var, from_=20, to=100, orient=tk.HORIZONTAL).grid(row=3, column=1, padx=10, pady=5)

        self.height_var = tk.IntVar(value=50)
        ttk.Label(self, text="Grid Height: ").grid(row=3, column=2, padx=10, pady=5)        
        tk.Scale(self, variable=self.height_var, from_=20, to=100, orient=tk.HORIZONTAL).grid(row=3, column=3, padx=10, pady=5)

        #Labels and Entries for parameters
        ttk.Label(self, text= "instance duration (s): ").grid(row=4, column=0, padx=10, pady=5)
        ttk.Entry(self, textvariable=self.grain_duration_var).grid(row=4, column=1, padx=10, pady=5)

        # New checkbox to enable or disable random duration
        ttk.Checkbutton(self, text="Random Instance Duration", variable=self.random_duration_var).grid(row=5, column=0, padx=10, pady=5)

        ttk.Label(self, text="Min Random Duration: ").grid(row=5, column=1, padx=10, pady=5)
        Scale(self, variable=self.min_random_duration_var, from_=0.1, to=10.0, resolution=0.01, orient=HORIZONTAL).grid(row=6, column=1, padx=10, pady=5)

        ttk.Label(self, text="Max Random Duration: ").grid(row=5, column=2, padx=10, pady=5)
        Scale(self, variable=self.max_random_duration_var, from_=0.1, to=10.0, resolution=0.01, orient=HORIZONTAL).grid(row=6, column=2, padx=10, pady=5)

        ttk.Label(self, text="Fixed Time Span (s): ").grid(row=7, column=0, padx=10, pady=5)
        ttk.Entry(self, textvariable=self.fixed_time_span_var).grid(row=7, column=1, padx=10, pady=5)

        ttk.Checkbutton(self, text="Random Time Span", variable=self.random_time_span_var).grid(row=8, column=0, padx=10, pady=5)

        ttk.Label(self, text="Min Random Time Span: ").grid(row=8, column=1, padx=10, pady=5)
        Scale(self, variable=self.min_random_time_span_var, from_=0.1, to=5.0, resolution=0.01, orient=HORIZONTAL).grid(row=9, column=1, padx=10, pady=5)

        ttk.Label(self, text="Max Random Time Span: ").grid(row=8, column=2, padx=10, pady=5)
        Scale(self, variable=self.max_random_time_span_var, from_=0.1, to=5.0, resolution=0.01, orient=HORIZONTAL).grid(row=9, column=2, padx=10, pady=5)

        ttk.Label(self, text="# frames per instance: ").grid(row=10, column=0, padx=10, pady=5)
        ttk.Entry(self, textvariable=self.num_generations_var).grid(row=10, column=1, padx=10, pady=5)

        ttk.Label(self, text="# of instances: ").grid(row=11, column=0, padx=10, pady=5)
        #tk.Scale(self, from_=1, to=30, orient=tk.HORIZONTAL, variable=self.num_instances_var, command=self.on_num_instances_change).grid(row=11, column=1, padx=10, pady=5)

        self.num_instances_slider = tk.Scale(self, from_=1, to=30, orient=tk.HORIZONTAL, variable=self.num_instances_var, command=self.on_num_instances_slider_change)
        self.num_instances_slider.grid(row=11, column=1, padx=10, pady=5)

        # Amplitude Mapping Dropdown
        self.amp_mapping_var = tk.StringVar(value="Default")
        ttk.Label(self, text="Amplitude Mapping: ").grid(row=12, column=0, padx=10, pady=5)
        amp_mapping_dropdown = ttk.Combobox(self, textvariable=self.amp_mapping_var, values=(
            "Default", "Linear Mapping", "Non-linear Mapping",
            "Conditional Mapping based on CA State", "Random Variability"))
        amp_mapping_dropdown.grid(row=12, column=1, padx=10, pady=5)

        # Stereo Pan Method Dropdown
        self.pan_method_var = tk.StringVar(value="Default")
        ttk.Label(self, text="Pan Method: ").grid(row=12, column=2, padx=10, pady=5)
        pan_method_dropdown = ttk.Combobox(self, textvariable=self.pan_method_var, values=(
            "Default", "Conditional Mapping based on CA State", "Non-linear Mapping", 
            "Random Variability", "Coordinate-Based Mapping"))
        pan_method_dropdown.grid(row=12, column=3, padx=10, pady=5)

        # Waveform Selection Dropdown
        ttk.Label(self, text="Waveform: ").grid(row=13, column=0, padx=10, pady=5)
        self.waveform_var = tk.StringVar(value="Sine")
        waveform_dropdown = ttk.Combobox(self, textvariable=self.waveform_var, values=(
            "Sine", "Triangle", "Sawtooth", "Square", "Reverse Sawtooth", "Pulse", "White Noise"))
        waveform_dropdown.grid(row=13, column=1, padx=10, pady=5)

        # Sliders for setting min and max frequency
        ttk.Label(self, text="Min Frequency (Hz): ").grid(row=14, column=0, padx=10, pady=5)
        tk.Scale(self, variable=self.min_freq_var, from_=60, to=3600, orient=tk.HORIZONTAL).grid(row=14, column=1, padx=10, pady=5)

        ttk.Label(self, text="Max Frequency (Hz): ").grid(row=14, column=2, padx=10, pady=5)
        tk.Scale(self, variable=self.max_freq_var, from_=60, to=3600, orient=tk.HORIZONTAL).grid(row=14, column=3, padx=10, pady=5)

        # Frequency Mapping Dropdown
        ttk.Label(self, text="Frequency Mapping: ").grid(row=15, column=0, padx=10, pady=5)
        self.freq_mapping_var = tk.StringVar(value="Standard")
        self.freq_mapping_dropdown = ttk.Combobox(self, textvariable=self.freq_mapping_var, values=(
            "Standard", "Complex Frequency Mapping", "CA Rule-Based Frequency", "Frequency Interpolation", "Harmonics and Overtones"),
            state="readonly")
        self.freq_mapping_dropdown.grid(row=15, column=1, padx=10, pady=5)
        self.freq_mapping_dropdown.bind('<<ComboboxSelected>>', self.update_freq_mapping_ui)

        # Create UI for "Number of Harmonics" and "Detune Amount" but do not grid them yet
        self.harmonics_label = ttk.Label(self, text="Number of Harmonics: ")
        self.harmonics_var = tk.IntVar(value=1)
        self.harmonics_scale = tk.Scale(self, variable=self.harmonics_var, from_=1, to=10, orient=tk.HORIZONTAL)

        self.detune_label = ttk.Label(self, text="Detune Amount: ")
        self.detune_var = tk.DoubleVar(value=1.0)
        self.detune_scale = tk.Scale(self, variable=self.detune_var, from_=0.1, to=0.99, resolution=0.01, orient=tk.HORIZONTAL)
        
        # UI for the Synthesis Techniques
        synthesis_technique_dropdown = ttk.Combobox(self, textvariable=self.synthesis_technique_var, 
            values=("None", "Ring Modulation", "Amplitude Modulation", "Frequency Modulation"))
        synthesis_technique_dropdown.grid(row=17, column=0, padx=10, pady=5)
        synthesis_technique_dropdown.bind('<<ComboboxSelected>>', self.update_synthesis_ui)
        ttk.Label(self, text="Synthesis Technique: ").grid(row=17, column=0, padx=10, pady=5)

        # Initialize UI elements for Amplitude Modulation parameters but do not grid them yet
        self.am_mod_freq_label = ttk.Label(self, text="AM Frequency (Hz):")
        self.am_mod_freq_var = tk.DoubleVar(value=440)
        self.am_mod_freq_scale = tk.Scale(self, variable=self.am_mod_freq_var, from_=20, to=2000, orient=tk.HORIZONTAL)

        # Initialize UI elements for Ring Modulation parameters but do not grid them yet
        self.rm_freq_label = ttk.Label(self, text="Ring Mod. Frequency (Hz):")
        self.rm_freq_var = tk.DoubleVar(value=440)
        self.rm_freq_scale = tk.Scale(self, variable=self.rm_freq_var, from_=20, to=2000, orient=tk.HORIZONTAL)

        # Initialize UI elements for Frequency Modulation parameters but do not grid them yet
        self.fm_mod_index_label = ttk.Label(self, text="FM Modulation Index:")
        self.fm_mod_index_var = tk.DoubleVar(value=2.5)
        self.fm_mod_index_scale = tk.Scale(self, variable=self.fm_mod_index_var, from_=0, to=20, orient=tk.HORIZONTAL)

        self.fm_freq_label = ttk.Label(self, text="FM Frequency (Hz):")
        self.fm_freq_var = tk.DoubleVar(value=440)
        self.fm_freq_scale = tk.Scale(self, variable=self.fm_freq_var, from_=20, to=2000, orient=tk.HORIZONTAL)

        # Add filter selection UI
        ttk.Label(self, text="Filter Type:").grid(row=19, column=0, padx=5, pady=5)
        self.filter_type_var = tk.StringVar(value="LowPass")
        self.filter_type_combobox = ttk.Combobox(self, textvariable=self.filter_type_var, 
                                                 values=("LowPass", "HighPass", "BandPass"), state="readonly")
        self.filter_type_combobox.grid(row=19, column=1, padx=5, pady=5)

        # Add cutoff frequency slider
        ttk.Label(self, text="Cutoff Frequency:").grid(row=19, column=2, padx=5, pady=5)
        self.cutoff_freq_var = tk.DoubleVar(value=1000)
        self.cutoff_freq_slider = tk.Scale(self, variable=self.cutoff_freq_var, from_=20, to=20000, orient="horizontal")
        self.cutoff_freq_slider.grid(row=19, column=3, padx=5, pady=5)

        # Add Q factor slider
        ttk.Label(self, text="Q Factor:").grid(row=19, column=4, padx=5, pady=5)
        self.q_factor_var = tk.DoubleVar(value=0.707)  # Typical value for a Butterworth filter
        self.q_factor_slider = tk.Scale(self, variable=self.q_factor_var, from_=0.1, to=10.0, resolution=0.01, orient="horizontal")
        self.q_factor_slider.grid(row=19, column=5, padx=5, pady=5)

        # Dropdown selector for audio descriptors
        self.descriptor_var = tk.StringVar(value="Spectral Centroid")
        self.descriptor_options = [
            "Spectral Centroid", "Spectral Standard Deviation", "Spectral Skewness",
            "Spectral Kurtosis", "RMS Amplitude", "Spectral Flatness",
            "Spectral Noisiness", "Spectral Irregularity", "Spectral Flux",
            "Spectral Inharmonicity", "Temporal Centroid"
        ]
        #ttk.Label(self, text="Y-axis Descriptor:").grid(row=20, column=0, padx=10, pady=5)
        ttk.Combobox(self, textvariable=self.descriptor_var, values=self.descriptor_options, state="readonly").grid(row=20, column=0, padx=10, pady=5)
        
        self.descriptor_var.trace_add('write', self.update_figure_with_points)

        # Add Reset Points Button
        ttk.Button(self, text="Reset Points", command=self.reset_drawn_points).grid(row=20, column=2, padx=10, pady=10)

        # Add the "Done" button
        ttk.Button(self, text="Done", command=self.on_done_button_clicked).grid(row=21, column=2, padx=10, pady=10, sticky="ew")

        # Matplotlib Figure
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.plot = self.fig.add_subplot(111)
        self.plot.set_title('Dynamic Points', fontsize=11)
        self.plot.set_xlim(0, 10)  # Adjust based on your expected number of instances
        self.plot.set_ylim(0, 1)  # Y-axis range from 0 to 1

        self.canvas = FigureCanvasTkAgg(self.fig, self)  # Embed figure in Tkinter
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=20, column=1, padx=10, pady=10)

        self.drawn_points = []  # Initialize an empty list to store points
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        self.dragging_point = None  # Track if a point is being dragged

        # Connect event handlers for dragging points
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)

        # Use self.dynamic_param_frame.scrollable_frame to add widgets that should be inside the scrollable area

        # Call it once initially to set up the default view
        self.update_synthesis_ui()     

    def on_num_instances_change(self, value):
        # Adjust existing points based on the new number of instances
        num_instances = int(value)  # Convert the slider value to an integer
        self.drawn_points = self.drawn_points[:num_instances]  # Keep only up to num_instances points
        # Evenly distribute points if fewer than num_instances
        if len(self.drawn_points) < num_instances:
            existing_x_vals = [point[0] for point in self.drawn_points]
            all_possible_x_vals = set(range(1, num_instances + 1))
            missing_x_vals = list(all_possible_x_vals - set(existing_x_vals))
            for x in missing_x_vals:
                self.drawn_points.append((x, 0.5))  # Add missing points with default y=0.5
        self.update_figure_with_points()

    def on_num_instances_slider_change(self, event=None):
        self.drawn_points.clear()  # Clear existing points
        self.update_figure_with_points()  # Redraw the canvas without points

    def on_canvas_click(self, event):
        if event.inaxes != self.plot.axes: return
        # Check if we already have maximum points
        if len(self.drawn_points) < self.num_instances_var.get():
            # Adjust to ensure points are within the new x-axis limits
            closest_x = round(event.xdata)
            # Limit closest_x to the range [1, num_instances]
            closest_x = max(1, min(self.num_instances_var.get(), closest_x))
            # Update or add the new point, ensuring unique x values
            existing_x_vals = [point[0] for point in self.drawn_points]
            if closest_x in existing_x_vals:
                idx = existing_x_vals.index(closest_x)
                self.drawn_points[idx] = (closest_x, event.ydata)
            else:
                self.drawn_points.append((closest_x, event.ydata))
            self.update_figure_with_points()
        else:
            print("Maximum number of points reached.")

    def reset_drawn_points(self):
        """Clears all drawn points and updates the figure."""
        self.drawn_points.clear()  # Clear the list of drawn points
        self.update_figure_with_points()  # Redraw the canvas without points

    def update_figure_with_points(self):
        # Clear the current plot
        self.plot.clear()
        self.plot.set_xlim(0, self.num_instances_var.get())
        self.plot.set_ylim(0, 1)
        if self.drawn_points:
            x_vals, y_vals = zip(*self.drawn_points)
            self.plot.plot(x_vals, y_vals, '-o')  # Redraw with updated positions
        self.canvas.draw()

    def clear_points_and_update_canvas(self, *args):
        self.drawn_points.clear()  # Clear the list of points
        self.update_figure_with_points()  # Redraw the canvas without points

    def on_mouse_press(self, event):
        # Check if any point is close to the mouse click position
        for idx, (x, y) in enumerate(self.drawn_points):
            if abs(event.xdata - x) < 0.1 and abs(event.ydata - y) < 0.1:  # Threshold to consider close
                self.dragging_point = idx
                return

    def on_mouse_move(self, event):
        # If dragging a point, update its position
        if self.dragging_point is not None and event.xdata is not None and event.ydata is not None:
            x, _ = self.drawn_points[self.dragging_point]
            self.drawn_points[self.dragging_point] = (x, event.ydata)  # Update y-position, keep x fixed
            self.update_figure_with_points()

    def on_mouse_release(self, event):
        # Stop dragging
        self.dragging_point = None            

    def interpolate_breakpoints(self, num_instances):
        """Interpolate the drawn points across the specified number of instances."""
        if not self.drawn_points:
            print("No points drawn.")
            return

        x_vals, y_vals = zip(*sorted(self.drawn_points))  # Sort points based on x values
        new_x_vals = np.linspace(min(x_vals), max(x_vals), num=num_instances)
        interpolated_y_vals = np.interp(new_x_vals, x_vals, y_vals)
        
        # Update synthesis parameters based on interpolated values
        self.update_synthesis_parameters(interpolated_y_vals)

    def update_synthesis_parameters(self, interpolated_values):
        """
        Update synthesis parameters based on interpolated y-values.
        For simplicity, this example directly maps interpolated values to the FM Modulation Index.
        """

        # Assume `interpolated_values` directly corresponds to the desired FM Modulation Indexes for each instance
        fm_modulation_indexes = interpolated_values

        # Example: Print the FM Modulation Index for each instance
        for i, value in enumerate(fm_modulation_indexes, start=1):
            print(f"Instance {i}: FM Modulation Index = {value}")

        # Update the actual synthesis engine parameters
        # This requires a callback function that can update the synthesis parameters outside of this class
        if callable(self.update_synynthesis_callback):
            self.update_synthesis_callback(fm_modulation_indexes)
        else:
            print("No callback function provided to update synthesis parameters.")

    def update_synthesis_ui(self, event=None):
        # Hide all parameters first
        for widget in [self.am_mod_freq_label, self.am_mod_freq_scale, self.rm_freq_label, 
                       self.rm_freq_scale, self.fm_mod_index_label, self.fm_mod_index_scale,
                       self.fm_freq_label, self.fm_freq_scale]:
            widget.grid_forget()

        # Based on the selection, grid (show) specific parameters
        if self.synthesis_technique_var.get() == "Amplitude Modulation":
            self.am_mod_freq_label.grid(row=17, column=1, padx=10, pady=5)
            self.am_mod_freq_scale.grid(row=17, column=2, padx=10, pady=5)
        elif self.synthesis_technique_var.get() == "Ring Modulation":
            self.rm_freq_label.grid(row=17, column=1, padx=10, pady=5)
            self.rm_freq_scale.grid(row=17, column=2, padx=10, pady=5)
        elif self.synthesis_technique_var.get() == "Frequency Modulation":
            self.fm_mod_index_label.grid(row=17, column=1, padx=10, pady=5)
            self.fm_mod_index_scale.grid(row=17, column=2, padx=10, pady=5)
            self.fm_freq_label.grid(row=17, column=3, padx=10, pady=5)
            self.fm_freq_scale.grid(row=17, column=4, padx=10, pady=5)  

    def update_freq_mapping_ui(self, event=None):
        if self.freq_mapping_var.get() == "Harmonics and Overtones":
            # Grid (show) the harmonics and detune options
            self.harmonics_label.grid(row=16, column=0, padx=10, pady=5)
            self.harmonics_scale.grid(row=16, column=1, padx=10, pady=5)
            self.detune_label.grid(row=16, column=2, padx=10, pady=5)
            self.detune_scale.grid(row=16, column=3, padx=10, pady=5)
        else:
            # Forget (hide) the harmonics and detune options
            self.harmonics_label.grid_forget()
            self.harmonics_scale.grid_forget()
            self.detune_label.grid_forget()
            self.detune_scale.grid_forget()

    # Method to apply the selected filter to the audio
    def apply_filter_with_q_factor(self, audio, sample_rate=44100):
        """
        Aplica filtros de áudio com validações robustas para evitar erros
        """
        # Validação básica de entrada
        if audio is None or audio.size == 0:
            print("Warning: Empty audio signal, returning original")
            return audio
        
        # Garantir que o áudio seja 2D (estéreo)
        if audio.ndim == 1:
            audio = np.column_stack((audio, audio))
        
        # Verificar tamanho mínimo para filtragem
        min_samples_required = 20
        if audio.shape[0] < min_samples_required:
            print(f"Warning: Audio too short for filtering ({audio.shape[0]} samples < {min_samples_required}), returning original")
            return audio
        
        # Obter parâmetros do filtro
        filter_type = self.filter_type_var.get()
        cutoff_freq = self.cutoff_freq_var.get()
        q_factor = self.q_factor_var.get()
        
        # Validações de frequência
        nyquist = sample_rate / 2.0
        if cutoff_freq <= 0:
            print("Warning: Invalid cutoff frequency, returning original audio")
            return audio
        
        if cutoff_freq >= nyquist:
            cutoff_freq = nyquist * 0.95  # Limite seguro
            print(f"Warning: Cutoff frequency too high, adjusted to {cutoff_freq:.1f} Hz")
        
        try:
            if filter_type.lower() == 'bandpass' and q_factor > 0:
                # Cálculo da largura de banda baseado no fator Q
                bandwidth = cutoff_freq / q_factor
                low_freq = max(1, cutoff_freq - bandwidth / 2.0)  # Mínimo de 1 Hz
                high_freq = min(nyquist * 0.95, cutoff_freq + bandwidth / 2.0)
                
                # Validar range do bandpass
                if low_freq >= high_freq:
                    print("Warning: Invalid bandpass range, returning original audio")
                    return audio
                
                # Normalizar frequências
                low_norm = low_freq / nyquist
                high_norm = high_freq / nyquist
                
                # Verificar se as frequências normalizadas são válidas
                if low_norm <= 0 or high_norm >= 1 or low_norm >= high_norm:
                    print("Warning: Invalid normalized frequencies for bandpass, returning original audio")
                    return audio
                
                sos = butter(N=2, Wn=[low_norm, high_norm], btype='bandpass', output='sos')
                
            else:
                # Para filtros passa-baixa e passa-alta
                normalized_cutoff = cutoff_freq / nyquist
                
                # Validar frequência normalizada
                if normalized_cutoff <= 0 or normalized_cutoff >= 1:
                    print("Warning: Invalid normalized cutoff frequency, returning original audio")
                    return audio
                
                sos = butter(N=2, Wn=normalized_cutoff, btype=filter_type.lower(), output='sos')

            # Aplicar filtro canal por canal para evitar problemas com arrays 2D
            filtered_audio = np.zeros_like(audio)
            
            for channel in range(audio.shape[1]):
                channel_data = audio[:, channel]
                
                # Verificar se o canal tem dados válidos
                if np.all(channel_data == 0):
                    filtered_audio[:, channel] = channel_data
                    continue
                
                # Aplicar filtro
                try:
                    filtered_channel = sosfiltfilt(sos, channel_data)
                    filtered_audio[:, channel] = filtered_channel
                except Exception as e:
                    print(f"Warning: Filter failed on channel {channel}: {e}")
                    filtered_audio[:, channel] = channel_data  # Manter original se falhar
            
            return filtered_audio
            
        except Exception as e:
            print(f"Filter error: {e}, returning original audio")
            return audio

    def on_parameter_change(self, event=None):
        # Set a flag or regenerate audio directly if performance allows
        self.flag_parameters_changed = True

    def create_or_continue_space(self, instance_num, last_space):
        # Retrieve the current width and height from the GUI sliders
        grid_width = self.width_var.get()
        grid_height = self.height_var.get()

        # Create a new grid with the specified dimensions
        space = np.zeros((self.num_generations_var.get(), grid_width, grid_height))

        # Fill the grid based on the initial conditions
        if instance_num == 0 or last_space is None:
            init_grid_percent = self.init_grid_percent_var.get()
            space[0] = np.random.choice([0, 1], size=(grid_width, grid_height), p=[1 - init_grid_percent, init_grid_percent])
        else:
            # If continuing from the last state, make sure to handle dimension changes
            last_state = last_space[-1]
            min_width = min(grid_width, last_state.shape[0])
            min_height = min(grid_height, last_state.shape[1])
            space[0][:min_width, :min_height] = last_state[:min_width, :min_height]

        return space

    def frequency_mapping(self, x, y, generation):
        freq_mapping_choice = self.freq_mapping_var.get()
        gen_width, gen_height = self.width_var.get(), self.height_var.get()

        # Get min and max frequency from the sliders
        min_freq = self.min_freq_var.get()
        max_freq = self.max_freq_var.get()

        # Initialize freq with a default value
        freq = 440  # Default frequency (A4 note)

        if freq_mapping_choice == "Frequency Interpolation":
            freq = self.interpolate_frequency(x, y, generation)
        elif freq_mapping_choice == "Complex Frequency Mapping":
            # Complex frequency mapping based on the number of alive neighbors
            neighbors = np.sum(generation[max(0, x-1):min(x+2, gen_width), max(0, y-1):min(y+2, gen_height)]) - generation[x, y]
            freq = np.interp(neighbors, [0, 8], [min_freq, max_freq])  # Map number of neighbors to frequency range
        elif freq_mapping_choice == "CA Rule-Based Frequency":
            # Frequency modulation based on the number of alive neighbors
            neighbors = np.sum(generation[max(0, x-1):min(x+2, gen_width), max(0, y-1):min(y+2, gen_height)]) - generation[x, y]
            freq = np.interp(neighbors, [0, 8], [min_freq, max_freq])  # Example frequency range from min_freq to max_freq
        else:
            # Standard frequency mapping
            freq_x = np.interp(x, [0, gen_width - 1], [min_freq, max_freq])
            freq_y = np.interp(y, [0, gen_height - 1], [min_freq, max_freq])
            freq = np.sqrt(freq_x * freq_y)

        return freq

    def base_frequency_mapping(self, x, y, generation):
        gen_width, gen_height = self.width_var.get(), self.height_var.get()
        min_freq = self.min_freq_var.get()
        max_freq = self.max_freq_var.get()
        
        freq_x = np.interp(x, [0, gen_width - 1], [min_freq, max_freq])
        freq_y = np.interp(y, [0, gen_height - 1], [min_freq, max_freq])
        return np.sqrt(freq_x * freq_y)

    def interpolate_frequency(self, x, y, generation):
        gen_width, gen_height = self.width_var.get(), self.height_var.get()
        neighbors = [(nx, ny) for nx in range(x-1, x+2) for ny in range(y-1, y+2)
                    if 0 <= nx < gen_width and 0 <= ny < gen_height and (nx != x or ny != y)]

        neighbor_frequencies = [self.base_frequency_mapping(nx, ny, generation) for nx, ny in neighbors]
        if not neighbor_frequencies:
            return self.base_frequency_mapping(x, y, generation)

        avg_neighbor_freq = sum(neighbor_frequencies) / len(neighbor_frequencies)
        return avg_neighbor_freq

    def generate_harmonics(self, base_freq, num_harmonics, detune, duration, waveform_type):
        num_samples = int(duration * SAMPLE_RATE)
        harmonics = np.zeros((num_samples, 2))
        for n in range(1, num_harmonics + 1):
            harmonic_freq = base_freq * n * detune
            grain = self.generate_waveform(harmonic_freq, duration, waveform_type)
            harmonics += np.column_stack((grain, grain))  # Ensuring a 2D array
        return harmonics / num_harmonics

    def generate_waveform(self, freq, duration, waveform_type, pulse_duty_cycle=0.5):
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        if waveform_type == "Sine":
            return np.sin(2 * np.pi * freq * t)
        elif waveform_type == "Square":
            return scipy.signal.square(2 * np.pi * freq * t)
        elif waveform_type == "Sawtooth":
            return scipy.signal.sawtooth(2 * np.pi * freq * t)
        elif waveform_type == "Triangle":
            return scipy.signal.sawtooth(2 * np.pi * freq * t, 0.5)
        elif waveform_type == "Reverse Sawtooth":
            return scipy.signal.sawtooth(2 * np.pi * freq * t, width=1)
        elif waveform_type == "Pulse":
            return scipy.signal.square(2 * np.pi * freq * t, duty=pulse_duty_cycle)
        elif waveform_type == "White Noise":
            return np.random.uniform(-1, 1, int(SAMPLE_RATE * duration))
        else:
            return np.sin(2 * np.pi * freq * t)  # Default to sine wave

    def amplitude_mapping(self, x, y, method, ca_state, generation=None):
        gen_width, gen_height = self.width_var.get(), self.height_var.get()
        if method == 'Default':
            amp_x = np.interp(x, [0, gen_width - 1], [0.1, 1])
            amp_y = np.interp(y, [0, gen_height - 1], [0.1, 1])
            return np.sqrt(amp_x * amp_y)
        elif method == 'Linear Mapping':
            return (x + y) / (gen_width + gen_height)
        elif method == 'Non-linear Mapping':
            return np.sin(x / gen_width) * np.sin(y / gen_height)
        elif method == 'Conditional Mapping based on CA State':
            return 1 if ca_state == 1 else 0.1
        #elif method == 'Frequency-Dependent Mapping':
            #if generation is not None:
                #freq = self.frequency_mapping(x, y, generation)
                #return np.log1p(freq) / np.log1p(np.sqrt(gen_width * gen_height))
            #else:
                #return 1  # Default amplitude if generation data is not available
        elif method == 'Random Variability':
            return np.random.uniform(0.1, 1)
        else:
            return 1  # Fallback to maximum amplitude

    def panning_mapping(self, x, y, method):
        gen_width, gen_height = self.width_var.get(), self.height_var.get()
        if method == "Default":
            return np.interp(x, [0, gen_width - 1], [-1, 1])
        elif method == "Conditional Mapping based on CA State":
            return 1 if y > gen_height // 2 else -1
        #elif method == "Frequency-Dependent Mapping":
            # Check if there are any generations available
            #if self.all_spaces:
                # Use the last generation for frequency mapping
                #last_generation = self.all_spaces[-1]
                #freq = self.frequency_mapping(x, y, last_generation)
            #else:
                # Default frequency if no generations are available
                #freq = 440
            #return np.interp(freq, [60, 3600], [-1, 1])
        elif method == "Non-linear Mapping":
            return np.sin(np.pi * x / (gen_width - 1))
        elif method == "Random Variability":
            return random.uniform(-1, 1)
        elif method == "Coordinate-Based Mapping":
            return np.interp(x, [0, gen_width - 1], [-1, 1]) * np.interp(y, [0, gen_height - 1], [-1, 1])
        
    def apply_fade_in_out(self, audio, max_fade_fraction=0.05):
        num_samples = audio.shape[0]
        fade_samples = int(SAMPLE_RATE * max_fade_fraction)  # Calculate fade duration in samples
        fade_samples = min(fade_samples, num_samples // 10)  # Ensure fade duration is no more than 10% of audio length for each fade-in and fade-out
        
        # Create fade-in and fade-out curves
        fade_in_curve = np.linspace(0, 1, fade_samples)
        fade_out_curve = np.linspace(1, 0, fade_samples)
        
        # Apply fade-in
        audio[:fade_samples] *= fade_in_curve[:, np.newaxis]  # Apply fade curve to both channels
        
        # Apply fade-out
        audio[-fade_samples:] *= fade_out_curve[:, np.newaxis]  # Apply fade curve to both channels

        return audio
    
    def add_rm_controls(self):
        # Ring Modulation Modulating Frequency
        self.rm_modulating_freq_var = tk.DoubleVar(value=440)  # Default frequency in Hz
        ttk.Label(self.dynamic_param_frame, text="Modulating Frequency (Hz):").grid(row=0, column=0, sticky="w")
        tk.Scale(self.dynamic_param_frame, variable=self.rm_modulating_freq_var, from_=20, to=2000, orient=tk.HORIZONTAL).grid(row=0, column=1, sticky="ew")

    def add_fm_controls(self):
        # FM Modulating Frequency
        self.fm_modulating_freq_var = tk.DoubleVar(value=2.0)  # Example default value
        ttk.Label(self.dynamic_param_frame, text="Modulating Frequency (Hz):").grid(row=0, column=0, sticky="w")
        tk.Scale(self.dynamic_param_frame, variable=self.fm_modulating_freq_var, from_=0.1, to=100, resolution=0.01, orient=tk.HORIZONTAL).grid(row=0, column=1, sticky="ew")
    
        # FM Modulation Index
        self.fm_modulation_index_var = tk.DoubleVar(value=2.5)  # Example default value
        ttk.Label(self.dynamic_param_frame, text="Modulation Index:").grid(row=0, column=4, sticky="w")
        tk.Scale(self.dynamic_param_frame, variable=self.fm_modulation_index_var, from_=0.1, to=20.0, resolution=0.01, orient=tk.HORIZONTAL).grid(row=0, column=5, sticky="ew")

    def add_am_controls(self):
        # Amplitude Modulation Modulating Frequency
        self.am_modulating_freq_var = tk.DoubleVar(value=440)  # Default frequency in Hz
        ttk.Label(self.dynamic_param_frame, text="Modulating Frequency (Hz):").grid(row=0, column=0, sticky="w")
        tk.Scale(self.dynamic_param_frame, variable=self.am_modulating_freq_var, from_=1, to=20, resolution=0.01, orient=tk.HORIZONTAL).grid(row=0, column=1, sticky="ew")

        # Amplitude Modulation Index (Depth)
        self.am_modulation_index_var = tk.DoubleVar(value=1.0)  # Default modulation index
        ttk.Label(self.dynamic_param_frame, text="Modulation Index:").grid(row=1, column=0, sticky="w")
        tk.Scale(self.dynamic_param_frame, variable=self.am_modulation_index_var, from_=0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL).grid(row=1, column=1, sticky="ew")

    def ring_modulation(self, signal, modulating_frequency, sample_rate):
        t = np.arange(len(signal)) / sample_rate
        modulating_signal = np.sin(2 * np.pi * modulating_frequency * t)
        return signal * modulating_signal

    def amplitude_modulation(self, signal, modulating_frequency, sample_rate, modulation_index=1.0):
        t = np.arange(len(signal)) / sample_rate
        carrier = np.sin(2 * np.pi * modulating_frequency * t)
        modulated_signal = (1 + modulation_index * carrier) * signal
        return modulated_signal

    def frequency_modulation(self, signal, modulating_frequency, modulation_index, sample_rate):
        t = np.arange(len(signal)) / sample_rate
        instantaneous_frequency = np.cumsum(modulation_index * np.sin(2 * np.pi * modulating_frequency * t))
        modulated_signal = np.sin(2 * np.pi * instantaneous_frequency)
        return modulated_signal * np.max(signal)  # Normalize to original signal's max amplitude
    
    def generation_to_audio(self, generation, instance_duration, amp_method, pan_method):
        gen_width, gen_height = generation.shape
        audio = np.zeros((int(SAMPLE_RATE * instance_duration), 2))  # Initialize a 2D array for stereo audio

        for x in range(gen_width):
            for y in range(gen_height):
                if generation[x, y] == 1:
                    # Calculate frequency for the current cell
                    freq = self.frequency_mapping(x, y, generation)
                    amp = self.amplitude_mapping(x, y, amp_method, generation[x, y])
                    pan = self.panning_mapping(x, y, pan_method)

                    # Generate a mono grain
                    grain_mono = self.generate_waveform(freq, instance_duration, self.waveform_var.get())

                    # Ensure the grain is stereo (2D array)
                    grain_stereo = np.column_stack((grain_mono, grain_mono)) * amp

                    # Generate the base waveform for the current cell
                    base_waveform = self.generate_waveform(freq, instance_duration, self.waveform_var.get())

                    # Apply selected synthesis technique
                    synthesis_technique = self.synthesis_technique_var.get()
                    if synthesis_technique == "Ring Modulation":
                        modulating_frequency = 30  # Example modulating frequency, adjust as needed
                        audio_grain = self.ring_modulation(base_waveform, modulating_frequency, SAMPLE_RATE)
                    elif synthesis_technique == "Amplitude Modulation":
                        modulating_frequency = 5  # Example modulating frequency, adjust as needed
                        audio_grain = self.amplitude_modulation(base_waveform, modulating_frequency, SAMPLE_RATE)
                    elif synthesis_technique == "Frequency Modulation":
                        modulating_frequency = 2  # Example modulating frequency, adjust as needed
                        modulation_index = 2.5  # Example modulation index, adjust as needed
                        audio_grain = self.frequency_modulation(base_waveform, modulating_frequency, modulation_index, SAMPLE_RATE)
                    else:
                        audio_grain = base_waveform  # No modulation applied

                    # Apply panning
                    pan_left = (1 - pan) * grain_stereo[:, 0]
                    pan_right = pan * grain_stereo[:, 1]
                    audio[:, 0] += pan_left
                    audio[:, 1] += pan_right

        # Apply normalization or any other post-processing if needed
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0:
            audio /= max_amplitude

        return audio

    def calculate_spectral_centroid(self, audio):
        """
        Calcula o Centroide Espectral com validações robustas
        """
        try:
            # Validação de entrada
            if audio is None or audio.size == 0:
                return 0.5  # Valor padrão
            
            # Garantir que seja mono
            if audio.ndim == 2 and audio.shape[1] == 2:
                audio_mono = np.mean(audio, axis=1)
            else:
                audio_mono = audio.flatten() if audio.ndim > 1 else audio

            # Validar tamanho mínimo
            if len(audio_mono) < 1024:
                return 0.5  # Valor padrão
                
            # Verificar se há sinal não-zero
            if np.all(audio_mono == 0):
                return 0.5  # Valor padrão para silêncio
                
            # Normalizar se necessário
            max_val = np.max(np.abs(audio_mono))
            if max_val > 0:
                audio_mono = audio_mono / max_val
            
            # Calcular centroide espectral
            centroid = librosa.feature.spectral_centroid(y=audio_mono, sr=SAMPLE_RATE)[0].mean()
            
            # Normalizar para 0-1
            min_centroid, max_centroid = 60, 6000
            normalized_centroid = (centroid - min_centroid) / (max_centroid - min_centroid)
            normalized_centroid = np.clip(normalized_centroid, 0, 1)
            
            return normalized_centroid
            
        except Exception as e:
            print(f"Error calculating spectral centroid: {e}")
            return 0.5  # Valor padrão em caso de erro
    
    def apply_descriptor_target(self, instance_audio, target_value, descriptor_type):
        """
        Aplica ajustes no áudio para atingir o valor alvo do descritor selecionado
        """
        if descriptor_type == "Spectral Centroid":
            return self.adjust_spectral_centroid(instance_audio, target_value)
        elif descriptor_type == "RMS Amplitude":
            return self.adjust_rms_amplitude(instance_audio, target_value)
        # Adicionar outros descritores conforme necessário
        return instance_audio
    
    def adjust_spectral_centroid(self, audio, target_normalized):
        """
        Ajusta o centroide espectral do áudio aplicando filtros com validações
        """
        # Validação de entrada
        if audio is None or audio.size == 0:
            print("Warning: Empty audio for spectral centroid adjustment")
            return audio
        
        # Verificar tamanho mínimo
        if audio.shape[0] < 1024:  # Librosa precisa de pelo menos 1024 amostras
            print("Warning: Audio too short for spectral centroid calculation")
            return audio
            
        try:
            current_centroid = self.calculate_spectral_centroid(audio)
        except Exception as e:
            print(f"Error calculating spectral centroid: {e}")
            return audio
        
        # Se já está próximo do alvo, retorna o áudio original
        tolerance = 0.05
        if abs(current_centroid - target_normalized) < tolerance:
            return audio
        
        # Converte o valor normalizado para frequência real
        min_freq, max_freq = 60, 6000
        target_freq = target_normalized * (max_freq - min_freq) + min_freq
        
        # Validar frequência alvo
        target_freq = max(60, min(6000, target_freq))
        
        # Aplicar filtro baseado no alvo
        try:
            if target_normalized > 0.5:
                # Alvo mais brilhante - aplica passa-alta
                self.filter_type_var.set("HighPass")
                self.cutoff_freq_var.set(max(100, target_freq * 0.7))
            else:
                # Alvo mais escuro - aplica passa-baixa
                self.filter_type_var.set("LowPass")
                self.cutoff_freq_var.set(min(8000, target_freq * 1.3))
            
            return self.apply_filter_with_q_factor(audio)
        except Exception as e:
            print(f"Error applying filter: {e}")
            return audio  
    
    def calculate_spectral_standard_deviation(self, audio):
        """
        Calculate the Spectral Standard Deviation of the audio signal.
        """
        # Ensure audio is mono
        if audio.shape[1] == 2:
            audio = np.mean(audio, axis=1)

        # Compute the power spectrum
        power_spectrum = np.abs(np.fft.fft(audio)) ** 2
        return np.std(power_spectrum)

    def calculate_spectral_skewness(self, audio):
        """
        Calculate the Spectral Skewness of the audio signal.
        """
        # Ensure audio is mono
        if audio.shape[1] == 2:
            audio = np.mean(audio, axis=1)

        # Compute the power spectrum
        power_spectrum = np.abs(np.fft.fft(audio)) ** 2
        return scipy.stats.skew(power_spectrum)

    def calculate_spectral_kurtosis(self, audio):
        """
        Calculate the Spectral Kurtosis of the audio signal.
        """
        # Ensure audio is mono
        if audio.shape[1] == 2:
            audio = np.mean(audio, axis=1)

        # Compute the power spectrum
        power_spectrum = np.abs(np.fft.fft(audio)) ** 2
        return scipy.stats.kurtosis(power_spectrum)
    
    def adjust_rms_amplitude(self, audio, target_rms):
        """
        Ajusta a amplitude RMS com validações
        """
        if audio is None or audio.size == 0:
            return audio
            
        try:
            current_rms = self.calculate_rms(audio)
            if current_rms > 0 and target_rms > 0:
                # Limitar amplificação para evitar distorção
                scale_factor = min(10.0, target_rms / current_rms)
                return audio * scale_factor
        except Exception as e:
            print(f"Error adjusting RMS: {e}")
        
        return audio 

    def calculate_rms(self, audio):
            """
            Calculate the Root Mean Square (RMS) of the audio signal.
            """
            if audio.shape[1] == 2:  # Check if audio is stereo
                audio = np.mean(audio, axis=1)  # Convert to mono
            rms = np.sqrt(np.mean(np.square(audio)))
            return rms

    def calculate_spectral_flatness(self, audio):
        """
        Calculate the Spectral Flatness of the audio signal.
        """
        # Ensure audio is mono for spectral flatness calculation
        if audio.shape[1] == 2:
            audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio

        # Compute the power spectrum
        power_spectrum = np.abs(np.fft.fft(audio_mono)) ** 2
        geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10)))  # Adding a small value to avoid log(0)
        arithmetic_mean = np.mean(power_spectrum)

        # Calculate and return spectral flatness
        spectral_flatness = geometric_mean / arithmetic_mean
        return spectral_flatness        

    def calculate_noisiness(self, audio):
        """
        Calculate the noisiness of the audio signal.
        """
        # Convert to mono if stereo
        if audio.shape[1] == 2:
            audio = np.mean(audio, axis=1)

        # FFT to get frequency domain representation
        freq_domain = np.fft.fft(audio)
        freq_amplitudes = np.abs(freq_domain)

        # Split frequencies into low and high bands
        mid_point = len(freq_amplitudes) // 2
        low_freq_band = freq_amplitudes[:mid_point]
        high_freq_band = freq_amplitudes[mid_point:]

        # Calculate energy in each band
        low_freq_energy = np.sum(low_freq_band ** 2)
        high_freq_energy = np.sum(high_freq_band ** 2)

        # Calculate noisiness as the ratio of high frequency energy to total energy
        noisiness = high_freq_energy / (low_freq_energy + high_freq_energy)
        return noisiness

    def calculate_spectral_irregularity(self, audio):
        """
        Calculate the spectral irregularity of the audio signal.
        """
        # Convert to mono if stereo
        if audio.shape[1] == 2:
            audio = np.mean(audio, axis=1)

        # FFT to get frequency domain representation
        freq_domain = np.fft.fft(audio)
        freq_amplitudes = np.abs(freq_domain)

        # Calculate differences between adjacent amplitudes
        irregularity = np.mean(np.abs(np.diff(freq_amplitudes)))
        return irregularity

    def calculate_spectral_entropy(self, audio):
        """
        Calculate the spectral entropy of the audio signal.
        """
        # Convert to mono if stereo
        if audio.shape[1] == 2:
            audio = np.mean(audio, axis=1)

        # FFT to get frequency domain representation
        freq_domain = np.fft.fft(audio)
        power_spectrum = np.abs(freq_domain) ** 2

        # Normalize power spectrum
        normalized_spectrum = power_spectrum / np.sum(power_spectrum)

        # Calculate entropy
        spectral_entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum + 1e-10))  # 1e-10 to avoid log(0)
        return spectral_entropy
    
    def calculate_spectral_flux(self, audio):
            """
            Calculate the Spectral Flux of the audio signal.
            """
            # Ensure audio is mono
            if audio.shape[1] == 2:
                audio = np.mean(audio, axis=1)
            
            # Compute the STFT (Short-Time Fourier Transform)
            stft = librosa.stft(audio)
            stft_magnitude = np.abs(stft)

            # Calculate the Spectral Flux
            flux = np.sqrt(np.sum(np.diff(stft_magnitude, axis=1) ** 2, axis=0))
            return np.mean(flux)

    def calculate_inharmonicity(self, audio):
        """
        Calculate the Inharmonicity of the audio signal.
        """
        # Ensure audio is mono
        if audio.shape[1] == 2:
            audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio

        # Estimate the fundamental frequency using the YIN algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(audio_mono, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'), sr=SAMPLE_RATE)

        # Compute the Short-Time Fourier Transform (STFT) and its magnitude
        D = librosa.stft(audio_mono)
        D_magnitude = np.abs(D)

        # Get peak frequencies at each frame using piptrack
        pitches, magnitudes = librosa.piptrack(S=D_magnitude, sr=SAMPLE_RATE)

        # Calculate inharmonicity
        deviations = []
        for t in range(pitches.shape[1]):
            if not voiced_flag[t]:
                continue
            harmonic_freqs = f0[t] * np.arange(1, 6)  # Considering first five harmonics
            for i, h_freq in enumerate(harmonic_freqs):
                if h_freq == 0:  # Skip zero frequency
                    continue
                # Find closest peak frequency to the harmonic frequency
                idx = np.argmin(np.abs(pitches[:, t] - h_freq))
                peak_freq = pitches[idx, t]
                deviation = abs(peak_freq - h_freq)
                deviations.append(deviation)

        # Calculate average deviation
        inharmonicity = np.mean(deviations) if deviations else 0
        return inharmonicity

    def calculate_temporal_centroid(self, audio):
        """
        Calculate the Temporal Centroid of the audio signal.
        """
        # Ensure audio is mono
        if audio.shape[1] == 2:
            audio = np.mean(audio, axis=1)
        
        # Compute the envelope of the signal
        envelope = np.abs(librosa.onset.onset_strength(y=audio, sr=SAMPLE_RATE))
        temporal_centroid = np.sum(np.arange(len(envelope)) * envelope) / np.sum(envelope)
        return temporal_centroid / SAMPLE_RATE  # Normalize by sample rate

    def process_ga_synthesis(self):
        # Assume each layer has its own set of targets and parameters
        # For demonstration, let's define some placeholders for GA parameters
        generations = 50  # Number of GA generations
        pop_size = 20  # Population size
        # Parameter space example: {parameter_name: [min_value, max_value]}
        param_space = {
            'modulation_frequency': [20, 2000],
            'modulation_index': [0.1, 10],
            # Add more synthesis parameters as required
        }
        # Placeholder for target centroids, should be dynamically obtained
        target_centroids = self.get_dynamic_target_centroids()

        # Execute GA for each target centroid
        for target_centroid in target_centroids:
            # Adjust run_genetic_algorithm to accept and work with the single target_centroid
            best_parameters, best_fitness = self.run_genetic_algorithm(
                generations, pop_size, param_space, target_centroid)
            
            # Use the best_parameters from GA to synthesize sound
            # This step depends on your synthesis method. As a placeholder:
            synthesized_sound = self.synthesize_sound(best_parameters)
            # Do something with the synthesized sound, e.g., mix it into the layer's audio stream
            self.mix_into_layer_audio(synthesized_sound)

    def adjust_to_target_spectral_centroid(self, target_centroid, instance_audio):
        """
        Adjust synthesis parameters to reach the target spectral centroid for the given audio instance.
        """
        actual_centroid = self.calculate_spectral_centroid(instance_audio)
        adjustment_step = 100  # Hz adjustment step for filter cutoff frequency
        iterations = 0
        max_iterations = 10
        threshold = 50  # Acceptable difference between target and actual centroid in Hz
        
        while abs(target_centroid - actual_centroid) > threshold and iterations < max_iterations:
            # Example adjustment logic: increase/decrease filter cutoff frequency
            # This is a placeholder; your actual synthesis and filtering logic will be more complex
            if actual_centroid < target_centroid:
                self.filter_cutoff += adjustment_step  # Increase cutoff frequency to brighten the sound
            else:
                self.filter_cutoff -= adjustment_step  # Decrease cutoff frequency to darken the sound

            # Regenerate the instance audio with the new parameters
            instance_audio = self.generate_instance_audio()  # Placeholder for your actual sound generation method
            actual_centroid = self.calculate_spectral_centroid(instance_audio)

            iterations += 1

        return instance_audio
    
    def get_target_centroid_for_instance(self, instance_index, min_centroid=1000, max_centroid=5000):
        """
        Retrieves the target spectral centroid for a specified instance based on user-defined y-values.

        Parameters:
        - instance_index: The index of the instance for which the target spectral centroid is requested.
        - min_centroid: The minimum value of the spectral centroid corresponding to a y-value of 0.
        - max_centroid: The maximum value of the spectral centroid corresponding to a y-value of 1.

        Returns:
        - The target spectral centroid for the specified instance. If the instance index is out of range,
          returns None or a default centroid value.
        """
        # Ensure there are drawn points and the instance index is within range
        if self.drawn_points and 0 <= instance_index < len(self.drawn_points):
            # Extract the y-value for the specified instance
            _, y_value = self.drawn_points[instance_index]
            
            # Scale the y-value to the spectral centroid range
            target_centroid = self.scale_target_centroid(y_value, min_centroid, max_centroid)
            return target_centroid
        else:
            # Instance index out of range or no points drawn; return None or consider a default value
            return (min_centroid + max_centroid) / 2  # Default to the midpoint of the centroid range as a fallback

    def scale_target_centroid(self, y_value, min_centroid, max_centroid):
        """
        Scale a y-value (ranging between 0 and 1) to a spectral centroid target value within a specified range.

        Parameters:
        - y_value: The normalized y-value obtained from user input, ranging from 0 to 1.
        - min_centroid: The minimum spectral centroid value, corresponding to y-value = 0.
        - max_centroid: The maximum spectral centroid value, corresponding to y-value = 1.

        Returns:
        - The scaled spectral centroid value.
        """
        # Ensure y_value is within bounds
        y_value = max(0, min(1, y_value))
        # Scale y_value to the target spectral centroid range
        return y_value * (max_centroid - min_centroid) + min_centroid
    
    def validate_points(self):
        """
        Valida e ajusta os pontos desenhados para corresponder ao número de instâncias
        """
        num_instances = self.num_instances_var.get()
        if len(self.drawn_points) > num_instances:
            self.drawn_points = self.drawn_points[:num_instances]
            self.update_figure_with_points()
        elif len(self.drawn_points) < num_instances:
            # Preenche pontos faltantes com valor padrão (0.5)
            for i in range(len(self.drawn_points), num_instances):
                self.drawn_points.append((i + 1, 0.5))
            self.update_figure_with_points()
    
    # E esta também dentro da classe CALayerTab:
    def get_target_value_for_instance(self, instance_num):
        """
        Obtém o valor alvo do descritor para uma instância específica
        """
        self.validate_points()  # Garante que há pontos suficientes
        
        if instance_num < len(self.drawn_points):
            _, y_value = self.drawn_points[instance_num]
            return y_value
        else:
            return 0.5  # Valor padrão se não houver ponto definido  

    def run_genetic_algorithm(self, generations, pop_size, param_space, all_layers_target_centroids):
        """
        Runs the genetic algorithm to find synthesis parameters that approach the target spectral centroids.

        :param generations: The number of generations to evolve.
        :param pop_size: The size of the population.
        :param param_space: The parameter space for the synthesis parameters.
        :param all_layers_target_centroids: A list of target spectral centroid values for each layer.
        :return: The best individual and its fitness from the last generation.
        """
        def initialize_population(pop_size, param_space):
            """
            Initializes a population for the GA.
            
            :param pop_size: Size of the population
            :param param_space: A dict defining the parameter spacegive me the complete
            :return: A list of dicts representing the population
            """
            population = []
            for _ in range(pop_size):
                individual = {param: random.choice(values) for param, values in param_space.items()}
                population.append(individual)
            return population
        
        def calculate_fitness(self, audio, y_value, sr=SAMPLE_RATE, min_centroid=60, max_centroid=6000):
            """
            Calculate the fitness of an audio instance based on its spectral centroid.

            Parameters:
            - audio: The audio signal (numpy array).
            - y_value: The y-value representing the target spectral centroid, scaled between 0 and 1.
            - sr: The sample rate of the audio signal.
            - min_centroid, max_centroid: The minimum and maximum spectral centroid values corresponding to y-value 0 and 1.

            Returns:
            - A fitness score indicating how close the audio's spectral centroid is to the target.
            """

            # Scale the y-value to the actual spectral centroid target
            target_centroid = self.scale_target_centroid(y_value, min_centroid, max_centroid)
            
            # Calculate the actual spectral centroid of the audio
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0].mean()

            # Calculate fitness as the inverse of the absolute difference between target and actual centroids
            # Note: You might want to scale/normalize this difference based on your specific needs
            fitness = 1 / (1 + abs(target_centroid - spectral_centroid))

            return fitness

        def tournament_selection(population, fitnesses, tournament_size=3):
            """
            Selects an individual from the population using tournament selection.
            
            :param population: The current population
            :param fitnesses: A list of fitness values corresponding to the population
            :param tournament_size: Number of individuals participating in each tournament
            :return: The winning individual of the tournament
            """
            participants = random.sample(list(zip(population, fitnesses)), tournament_size)
            winner = max(participants, key=lambda item: item[1])
            return winner[0]

        def crossover(parent1, parent2):
            """Performs a simple crossover between two parents to produce two children."""
            child1, child2 = parent1.copy(), parent2.copy()
            crossover_point = random.randint(1, len(parent1) - 2)  # Exclude first and last
            
            for i, param in enumerate(parent1):
                if i > crossover_point:
                    child1[param], child2[param] = child2[param], child1[param]
            return child1, child2

        def mutate(individual, param_space, mutation_rate=0.1):
            """Mutates an individual's parameters based on the mutation rate."""
            for param in individual:
                if random.random() < mutation_rate:
                    individual[param] = random.choice(param_space[param])

        # Initialize the population
        population = initialize_population(pop_size, param_space)

        for generation in range(generations):
            # Evaluate the fitness of each individual in the population
            fitnesses = [calculate_fitness(individual, all_layers_target_centroids) for individual in population]

            # Selection
            selected = [tournament_selection(population, fitnesses) for _ in range(pop_size)]

            # Generate new population through crossover and mutation
            new_population = []
            while len(new_population) < pop_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = crossover(parent1, parent2)
                mutate(child1, param_space)
                mutate(child2, param_space)
                new_population.extend([child1, child2])

            # Update population for the next generation
            population = new_population[:pop_size]  # Ensure population size stays constant

            # Logging progress
            if generation % 10 == 0 or generation == generations - 1:
                best_fitness = max(fitnesses)
                print(f"Generation {generation}: Best Fitness = {best_fitness}")

        # After all generations, find the best individual
        final_fitnesses = [calculate_fitness(individual, all_layers_target_centroids) for individual in population]
        best_index = final_fitnesses.index(max(final_fitnesses))
        best_individual = population[best_index]
        best_fitness = final_fitnesses[best_index]

        return best_individual, best_fitness

    # Note: The `calculate_fitness` function is assumed to accept an individual and a list of target spectral centroids.
    # The individual represents a set of synthesis parameters, and the function should return how well these parameters
    # meet the target spectral centroids (higher fitness for closer matches).

    def export_images(self, filename_prefix):
        dpi_value = 300  # For high resolution

        if not self.all_spaces:
            self.status_label["text"] = "No CA states to visualize."
            return

        for instance_num, generations in enumerate(self.all_spaces):
            print(f"Instance {instance_num+1}:")
            for gen_num, generation in enumerate(generations):
                print(f"  Generation {gen_num+1}: shape={generation.shape}, type={type(generation)}")
                # Check if 'generation' is a 2D array
                if generation.ndim == 2 and generation.size > 0:  # Check if it's a non-empty 2D array
                    fig, ax = plt.subplots()
                    ax.imshow(generation, cmap='binary', aspect='equal')
                    ax.axis('off')  # Hide the axis for a clean image

                    # Save the figure
                    frame_filename = f"{filename_prefix}_instance_{instance_num+1}_gen_{gen_num+1}.jpg"
                    plt.savefig(frame_filename, dpi=dpi_value, bbox_inches='tight')
                    plt.close(fig)
                else:
                    print(f"Generation {gen_num + 1} in Instance {instance_num + 1} is not 2D or is empty.")

        self.status_label["text"] = "Images exported successfully."

    def export_gifs(self, filename_prefix):
        try:
            if not self.all_spaces:
                raise ValueError("No CA states to visualize")

            for instance_num, generations in enumerate(self.all_spaces):
                print(f"Exporting GIF for instance {instance_num + 1}")
                
                if not isinstance(generations, np.ndarray) or generations.ndim != 3:
                    print(f"Error: Instance {instance_num + 1} has invalid data")
                    continue

                grid_width = self.width_var.get()
                grid_height = self.height_var.get()
                aspect_ratio = grid_width / grid_height
                frames = []
                
                import tempfile
                import os

                for frame_idx, gen in enumerate(generations):
                    try:
                        # Criar figura para cada frame
                        fig, ax = plt.subplots(figsize=(8 * aspect_ratio, 8), dpi=100)
                        ax.imshow(gen, cmap='binary', aspect='auto')
                        ax.set_title(f'2D Cellular Automaton (Instance {instance_num + 1})')
                        ax.axis('off')
                        
                        # Salvar como arquivo temporário
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                            fig.savefig(tmp_file.name, format='png', bbox_inches='tight', 
                                    facecolor='white', edgecolor='none', dpi=100)
                            plt.close(fig)
                            
                            # Ler a imagem como array numpy
                            temp_image = imageio.imread(tmp_file.name)
                            frames.append(temp_image)
                            
                            # Remover arquivo temporário
                            os.unlink(tmp_file.name)
                        
                    except Exception as e:
                        print(f"Error processing frame {frame_idx}: {e}")
                        plt.close('all')
                        continue

                # Salvar como GIF se há frames válidos
                if frames:
                    gif_filename = f'{filename_prefix}_instance_{instance_num + 1}.gif'
                    imageio.mimsave(gif_filename, frames, format='GIF', duration=0.2, loop=0)
                    print(f"GIF saved: {gif_filename}")
                else:
                    print(f"No valid frames for instance {instance_num + 1}")
            
            self.status_label["text"] = "GIF files exported successfully!"
            print("All GIFs exported successfully!")
            
        except Exception as e:
            print(f"Error exporting GIFs: {e}")
            self.status_label["text"] = f"Error exporting GIFs: {e}"
            plt.close('all')

    def export_videos_direct_method(self):
        """
        Método direto usando apenas matplotlib.pyplot.savefig
        """
        try:
            filename = self.filename_var.get()
            if not filename:
                raise ValueError("Filename is empty")

            # Definir backend sem GUI para evitar problemas
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            for i, tab in enumerate(self.main_gui.notebook.tabs()[2:], start=1):
                layer_tab = self.main_gui.notebook.nametowidget(tab)
                
                if not hasattr(layer_tab, 'all_spaces') or not layer_tab.all_spaces:
                    continue

                for instance_num, space in enumerate(layer_tab.all_spaces):
                    video_filename = f'{filename}_layer_{i}_instance_{instance_num+1}.mp4'
                    
                    # Criar lista de imagens
                    image_files = []
                    
                    for frame_idx, generation in enumerate(space):
                        # Nome do arquivo temporário
                        temp_filename = f"temp_frame_{frame_idx:04d}.png"
                        
                        # Criar e salvar figura
                        plt.figure(figsize=(10, 10), dpi=100)
                        plt.imshow(generation, cmap='binary', aspect='auto')
                        plt.title(f'Layer {i}, Instance {instance_num + 1}, Frame {frame_idx}')
                        plt.axis('off')
                        plt.savefig(temp_filename, bbox_inches='tight', facecolor='white')
                        plt.close()
                        
                        image_files.append(temp_filename)
                    
                    # Criar vídeo a partir das imagens
                    with imageio.get_writer(video_filename, fps=10) as writer:
                        for img_file in image_files:
                            image = imageio.imread(img_file)
                            writer.append_data(image)
                            os.remove(img_file)  # Remover arquivo temporário
                    
                    print(f"Video saved: {video_filename}")

            self.status_label["text"] = "Videos exported successfully!"
            
        except Exception as e:
            print(f"Error in direct video export: {e}")
            self.status_label["text"] = f"Error exporting video: {e}"            

    def rule(left, center, right, rule_number):
        rule_string = format(rule_number, '08b')
        configurations = ['111', '110', '101', '100', '011', '010', '001', '000']
        rule_dict = {config: int(rule_string[i]) for i, config in enumerate(configurations)}
        current_config = f"{int(left)}{int(center)}{int(right)}"
        return rule_dict[current_config]

    # Game of Life rules
    def GoA_next_generation(self, current_gen, gen_width, gen_height):
        new_gen = np.zeros((gen_width, gen_height))
        for x in range(1,gen_width - 1):
            for y in range(1, gen_height - 1):
                cell = current_gen[x, y]
                neighbors = np.sum(current_gen[x-1:x+2, y-1:y+2]) - cell
                if cell and (neighbors == 2 or neighbors == 3):
                    new_gen[x, y] = 1
                elif not cell and neighbors == 3:
                    new_gen[x, y] = 1
        return new_gen

    # New function to implement HighLife rules as an example
    def highlife_next_generation(self, current_gen, gen_width, gen_height):
        new_gen = np.zeros((gen_width, gen_height))
        for x in range(1, gen_width - 1):
            for y in range(1, gen_height - 1):
                cell = current_gen[x, y]
                neighbors = np.sum(current_gen[x-1:x+2, y-1:y+2]) - cell
                if cell and (neighbors == 2 or neighbors == 3):
                    new_gen[x, y] = 1
                elif not cell and (neighbors == 3 or neighbors == 6):
                    new_gen[x, y] = 1
        return new_gen

    def seeds_next_generation(self, current_gen, gen_width, gen_height):
        new_gen = np.zeros((gen_width, gen_height))
        for x in range(1, gen_width - 1):
            for y in range(1, gen_height - 1):
                cell = current_gen[x, y]
                neighbors = np.sum(current_gen[x-1:x+2, y-1:y+2]) - cell
                if not cell and neighbors == 2:
                    new_gen[x, y] = 1
        return new_gen

    def day_and_night_next_generation(self, current_gen, gen_width, gen_height):
        new_gen = np.zeros((gen_width, gen_height))
        for x in range(1, gen_width - 1):
            for y in range(1, gen_height - 1):
                cell = current_gen[x, y]
                neighbors = np.sum(current_gen[x-1:x+2, y-1:y+2]) - cell
                if cell and (neighbors == 3 or neighbors == 4 or neighbors == 6 or neighbors == 7 or neighbors == 8):
                    new_gen[x, y] = 1
                elif not cell and (neighbors == 3 or neighbors == 6 or neighbors == 7 or neighbors == 8):
                    new_gen[x, y] = 1
        return new_gen

    def anneal_next_generation(self, current_gen, gen_width, gen_height):
        new_gen = np.zeros((gen_width, gen_height))
        for x in range(1, gen_width - 1):
            for y in range(1, gen_height - 1):
                cell = current_gen[x, y]
                neighbors = np.sum(current_gen[x-1:x+2, y-1:y+2]) - cell
                if cell and (neighbors == 4 or neighbors == 5 or neighbors == 6 or neighbors == 7 or neighbors == 8):
                    new_gen[x, y] = 1
                elif not cell and (neighbors == 3 or neighbors == 6 or neighbors == 7 or neighbors == 8):
                    new_gen[x, y] = 1
        return new_gen

    def bacteria_next_generation(self, current_gen, gen_width, gen_height):
        new_gen = np.zeros((gen_width, gen_height))
        for x in range(1, gen_width - 1):
            for y in range(1, gen_height - 1):
                cell = current_gen[x, y]
                neighbors = np.sum(current_gen[x-1:x+2, y-1:y+2]) - cell
                if cell and (neighbors == 1 or neighbors == 5):
                    new_gen[x, y] = 1
                elif not cell and (neighbors == 5):
                    new_gen[x, y] = 1
        return new_gen

    def maze_next_generation(self, current_gen, gen_width, gen_height):
        new_gen = np.zeros((gen_width, gen_height))
        for x in range(1, gen_width - 1):
            for y in range(1, gen_height - 1):
                cell = current_gen[x, y]
                neighbors = np.sum(current_gen[x-1:x+2, y-1:y+2]) - cell
                if cell and (neighbors == 1 or neighbors == 2 or neighbors == 3 or neighbors == 4 or neighbors == 5):
                    new_gen[x, y] = 1
                elif not cell and (neighbors == 3):
                    new_gen[x, y] = 1
        return new_gen

    def coral_next_generation(self, current_gen, gen_width, gen_height):
        new_gen = np.zeros((gen_width, gen_height))
        for x in range(1, gen_width - 1):
            for y in range(1, gen_height - 1):
                cell = current_gen[x, y]
                neighbors = np.sum(current_gen[x-1:x+2, y-1:y+2]) - cell
                if cell and (neighbors == 4 or neighbors == 5 or neighbors == 6 or neighbors == 7 or neighbors == 8):
                    new_gen[x, y] = 1
                elif not cell and (neighbors == 3):
                    new_gen[x, y] = 1
        return new_gen

    def exploding_next_generation(self, current_gen, gen_width, gen_height):
        new_gen = np.zeros((gen_width, gen_height))
        for x in range(1, gen_width - 1):
            for y in range(1, gen_height - 1):
                cell = current_gen[x, y]
                neighbors = np.sum(current_gen[x-1:x+2, y-1:y+2]) - cell
                if cell and (neighbors == 0 or neighbors == 1 or neighbors == 8):
                    new_gen[x, y] = 1
                elif not cell and (neighbors == 3 or neighbors == 4 or neighbors == 5):
                    new_gen[x, y] = 1
        return new_gen

    # Function to interpolate between two CA states
    def interpolate_states(state1, state2, alpha):
        return state1 * (1 - alpha) + state2 * alpha
    
    def generate_audio_for_layer(self, use_ga=False):
        """
        Versão corrigida que integra os descritores de áudio
        """
        if use_ga:
            self.process_ga_synthesis()
        else:
            num_instances = self.num_instances_var.get()
            final_audio = np.array([])
            total_layer_duration = 0

            # Clear previous states to ensure a fresh start
            self.all_spaces = []

            # CA Generation Functions mapping
            ca_gen_func_map = {
                "Game of Life": self.GoA_next_generation,
                "HighLife": self.highlife_next_generation,
                "Seeds": self.seeds_next_generation,
                "Day and Night": self.day_and_night_next_generation,
                "Anneal": self.anneal_next_generation,
                "Bacteria": self.bacteria_next_generation,
                "Maze": self.maze_next_generation,
                "Coral": self.coral_next_generation,
                "Exploding Rules": self.exploding_next_generation
            }

            ca_type = self.ca_type_var.get()
            ca_gen_func = ca_gen_func_map.get(ca_type, None)
            if ca_gen_func is None:
                print(f"CA generation function for type '{ca_type}' not found.")
                return final_audio

            # Obter o descritor selecionado
            selected_descriptor = self.descriptor_var.get()

            for instance_num in range(num_instances):
                # Calculate instance duration
                if self.random_duration_var.get():
                    instance_duration = random.uniform(
                        self.min_random_duration_var.get(), 
                        self.max_random_duration_var.get()
                    )
                else:
                    instance_duration = self.grain_duration_var.get()

                frame_duration = instance_duration / self.num_generations_var.get()

                # Initialize or continue CA space
                space = self.create_or_continue_space(
                    instance_num, 
                    self.all_spaces[-1] if self.all_spaces else None
                )

                # Generate audio for each frame
                instance_audio = np.array([])
                for frame_num in range(self.num_generations_var.get()):
                    if frame_num > 0:
                        space[frame_num] = ca_gen_func(
                            space[frame_num - 1], 
                            self.width_var.get(), 
                            self.height_var.get()
                        )

                    frame_audio = self.generation_to_audio(
                        space[frame_num], 
                        frame_duration, 
                        self.amp_mapping_var.get(), 
                        self.pan_method_var.get()
                    )
                    frame_audio = self.apply_fade_in_out(frame_audio)
                    instance_audio = np.vstack([instance_audio, frame_audio]) if instance_audio.size else frame_audio

                # APLICAR DESCRITOR SE PONTOS FORAM DESENHADOS
                target_y_value = self.get_target_value_for_instance(instance_num)
                instance_audio = self.apply_descriptor_target(
                    instance_audio, 
                    target_y_value, 
                    selected_descriptor
                )

                # Normalize and append
                max_amplitude = np.max(np.abs(instance_audio))
                if max_amplitude > 0:
                    instance_audio /= max_amplitude
                final_audio = np.vstack([final_audio, instance_audio]) if final_audio.size else instance_audio

                self.all_spaces.append(space)
                total_layer_duration += instance_duration

                # Add silence between instances
                if instance_num < num_instances - 1:
                    if self.random_time_span_var.get():
                        time_span_duration = random.uniform(
                            self.min_random_time_span_var.get(), 
                            self.max_random_time_span_var.get()
                        )
                    else:
                        time_span_duration = self.fixed_time_span_var.get()
                    
                    silence = np.zeros((int(time_span_duration * SAMPLE_RATE), 2))
                    final_audio = np.vstack([final_audio, silence])
                    total_layer_duration += time_span_duration

            print(f'Total Duration for Layer {self.layer_num}: {total_layer_duration:.2f} seconds')
            return final_audio

    def adjust_grain_duration(self, instance_num, total_random_duration):
            if self.random_duration_var.get():
                min_random_duration = self.min_random_duration_var.get()
                max_random_duration = self.max_random_duration_var.get()
                GRAIN_DURATION = round(random.uniform(min_random_duration, max_random_duration), 2)
                total_random_duration += GRAIN_DURATION
                print(f"Instance {instance_num + 1} Duration: {GRAIN_DURATION} seconds")
            else:
                GRAIN_DURATION = self.grain_duration_var.get()
            return GRAIN_DURATION, total_random_duration

    def insert_silence(self, instance_num):
        # Calculate time span duration
        if self.random_time_span_var.get():
            min_random_time_span = self.min_random_time_span_var.get()
            max_random_time_span = self.max_random_time_span_var.get()
            time_span_duration = round(random.uniform(min_random_time_span, max_random_time_span), 2)
        else:
            time_span_duration = self.fixed_time_span_var.get()

        # Create a silence array with the calculated duration
        silence = np.zeros((int(time_span_duration * SAMPLE_RATE), 2))

        # Append silence to the audio
        if self.all_audio.size == 0:
            self.all_audio = silence
        else:
            self.all_audio = np.vstack([self.all_audio, silence])

        # Return the duration of the silence inserted
        return time_span_duration

    def modify_last_state(self, last_state):
        """
        Modifies the last state of the space by shifting rows or columns.
        """
        min_size = 2  # Ensure the grid is at least 2x2 to avoid ValueError in randint
        if last_state.shape[0] < min_size or last_state.shape[1] < min_size:
            return last_state  # Return unmodified state if too small to modify

        shift_mode = np.random.choice(['horizontal', 'vertical'])
        # Ensure shift_amount is always valid by limiting its range
        shift_amount = np.random.randint(1, max(2, min(last_state.shape[0], last_state.shape[1]) // 2))

        if shift_mode == 'horizontal':
            last_state = np.roll(last_state, shift=shift_amount, axis=1)
        elif shift_mode == 'vertical':
            last_state = np.roll(last_state, shift=shift_amount, axis=0)

        return last_state

    def create_or_continue_space(self, instance_num, last_space):
        grid_width = self.width_var.get()
        grid_height = self.height_var.get()
        
        # Initialize a new space for the current instance
        space = np.zeros((self.num_generations_var.get(), grid_width, grid_height))

        if instance_num == 0 or last_space is None:
            # For the first instance or if there's no previous state,
            # initialize the space randomly based on `init_grid_percent_var`.
            init_grid_percent = self.init_grid_percent_var.get()
            space[0] = np.random.choice([0, 1], size=(grid_width, grid_height), p=[1 - init_grid_percent, init_grid_percent])
        else:
            # If continuing from a previous state, ensure last_space is a list of 2D arrays.
            if isinstance(last_space, list) and len(last_space) > 0 and isinstance(last_space[-1], np.ndarray) and last_space[-1].ndim == 2:
                # Continue from the last state, possibly modifying it.
                modified_last_state = self.modify_last_state(last_space[-1].copy())
                # Ensure modified_last_state fits within the new space dimensions.
                min_width = min(grid_width, modified_last_state.shape[0])
                min_height = min(grid_height, modified_last_state.shape[1])
                space[0][:min_width, :min_height] = modified_last_state[:min_width, :min_height]
            else:
                # Handle unexpected last_space content (e.g., not a list of 2D arrays).
                print("Warning: Invalid last_space provided. Initializing first state randomly.")
                init_grid_percent = self.init_grid_percent_var.get()
                space[0] = np.random.choice([0, 1], size=(grid_width, grid_height), p=[1 - init_grid_percent, init_grid_percent])

        return space

    def process_space(self, space, last_space, GRAIN_DURATION):
        if last_space is not None:
            interpolated_space = self.interpolate_states(last_space[-1], space[0], 0.5)
            new_audio = self.generation_to_audio(interpolated_space, GRAIN_DURATION, self.amp_mapping_var.get(), self.pan_method_var.get())
            self.all_audio = np.vstack([self.all_audio, new_audio])

    def normalize_audio(self):
        max_val_left = np.max(np.abs(self.all_audio[:, 0]))
        max_val_right = np.max(np.abs(self.all_audio[:, 1]))

        if max_val_left > 0:
            self.all_audio[:, 0] /= max_val_left
        else:
            print("Warning: Left channel of audio contains only silence or very low values.")

        if max_val_right > 0:
            self.all_audio[:, 1] /= max_val_right
        else:
            print("Warning: Right channel of audio contains only silence or very low values.")

class CA_GUI:
    def __init__(self, master, root_window):  # Add root_window as a parameter
        self.master = master
        self.root_window = root_window  # Store the root window reference for other uses

        # Variables
        self.num_layers_var = tk.IntVar(value=1)

        # Create a Notebook widget (Tab controller)
        self.notebook = ttk.Notebook(master)
        self.notebook.grid(row=1, column=0, columnspan=4, padx=10, pady=10)

        # Create the fixed tabs
        self.layer_setting_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.layer_setting_tab, text="Layer Settings")

        self.global_controls_tab = GlobalControlsTab(self.notebook, self)  # Pass self here

        self.notebook.add(self.global_controls_tab, text="Global Controls")

        # Slider to adjust the number of layers
        ttk.Label(self.layer_setting_tab, text="Number of Layers: ").grid(row=0, column=0, padx=10, pady=5)
        tk.Scale(self.layer_setting_tab, variable=self.num_layers_var, from_=1, to=10, orient=tk.HORIZONTAL, 
                 command=self.update_tabs).grid(row=0, column=1, padx=10, pady=5)

        # Inside the Layer Settings tab setup in CA_GUI class
        ttk.Button(self.layer_setting_tab, text="Reset Settings", command=self.reset_settings).grid(row=1, column=0, padx=10, pady=10)

        # Collect list of Y-values in each Canva Figure and store them in a list
        ttk.Button(self.global_controls_tab, text="Collect Y-Values", command=self.collect_all_y_values).grid(row=1, column=2, padx=10, pady=10)

        self.update_tabs()

    def collect_all_y_values(self):
        all_y_values = []
        for i in range(2, len(self.notebook.tabs())):  # Skipping the fixed tabs
            layer_tab = self.notebook.nametowidget(self.notebook.tabs()[i])
            y_values = [y for _, y in layer_tab.drawn_points]
            all_y_values.append(y_values)
            print(f"Layer {i-1} y-values:", y_values)
        # Here, all_y_values is a list of lists containing y-values for each layer tab
        # Further processing can be done as needed

    def reset_settings(self):
        # Reset the number of layers to the default value
        self.num_layers_var.set(1)

        # Update the tabs based on the default number of layers
        self.update_tabs()

        # For example, resetting settings in each layer tab
        for tab in self.notebook.tabs()[2:]:  # Iterate through layer tabs
            layer_tab = self.notebook.nametowidget(tab)
            layer_tab.reset_layer_settings()  # Reset settings of each layer

    def update_synthesis_parameters(self, params):
        """
        Placeholder callback function for updating synthesis parameters.
        
        Parameters:
        - params: The parameters to update the synthesis with. This could be a dictionary,
                  a list, or any data structure that your synthesis logic requires.
        
        This method is designed to be called by CALayerTab instances when there is a need
        to update synthesis parameters based on user interaction or internal changes.
        """
        # Placeholder implementation: print the received parameters
        print("Received synthesis parameters for update:", params)
        
        # Here, you would add your logic to actually update the synthesis parameters.
        # For example, this could involve setting values on a synthesizer object,
        # adjusting global settings, or triggering a re-synthesis with the new parameters.

    def update_tabs(self, event=None):
        current_num_tabs = len(self.notebook.tabs())
        requested_num_layers = self.num_layers_var.get() + 2  # +2 for the fixed tabs

        # Remove extra tabs if the current number exceeds the requested number of layers
        while current_num_tabs > requested_num_layers:
            self.notebook.forget(current_num_tabs - 1)
            current_num_tabs -= 1

        # Add new tabs if the current number is less than the requested number of layers
        while current_num_tabs < requested_num_layers:
            layer_num = current_num_tabs - 1  # Adjust for the fixed tabs indexing
            new_tab = CALayerTab(self.notebook, layer_num, self.update_synthesis_parameters)
            self.notebook.add(new_tab, text=f'Layer {layer_num}')
            current_num_tabs += 1

        # Inside the CA_GUI class, in the update_tabs method, modify the call like so:
        for i in range(2, len(self.notebook.tabs())):  # Skip fixed tabs
            tab = self.notebook.nametowidget(self.notebook.tabs()[i])
            tab.update_figure_with_points()  # No arguments passed
    # Define the ScrollableFrame class
            
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

# Create main window
root = tk.Tk()
root.title("Multi-Layered Cellular Automaton Granular Synthesis")

# Set the window size based on the screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f'{screen_width}x{screen_height}')

# Create the scrollable frame and add it to the root window
main_scrollable_frame = ScrollableFrame(root)
main_scrollable_frame.pack(fill="both", expand=True)

# Instantiate your application, passing the scrollable frame's inner frame
# Assuming main_scrollable_frame.scrollable_frame is the content area and root is your tk.Tk() instance
app = CA_GUI(main_scrollable_frame.scrollable_frame, root_window=root)

# Start the application
root.mainloop()