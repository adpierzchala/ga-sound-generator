# GA Sound Generator

A Flask-based web application that uses a **genetic algorithm** (GA) to compose musical phrases. Users can specify tempo, instruments, phrase length, and even GA parameters to customize the generation process.

---

## Features

- **Specify Musical Parameters**:  
  Users can input:
  - Tempo (BPM)
  - Instruments (e.g., Piano, Violin, Flute) using checkboxes
  - Phrase length (number of notes)

- **Customize Genetic Algorithm Settings**:  
  Adjust the **number of generations** and **mutation rate** to explore different compositions.

- **Post-generation Actions**:  
  - Play the generated sound directly in the browser.  
  - Download the generated audio as a `.wav` file.  
  - View the generated notes and instruments used.

---

## Screenshots

### User Input Interface
![User Input](https://github.com/user-attachments/assets/f55a9269-03a0-4d2f-b1bb-2a567f461665)

### Parameters Configuration
![GA Parameters](https://github.com/user-attachments/assets/48ccceaa-e524-48bb-9ee0-5639803fe832)

### Post-generation Actions
![Post-generation Actions](https://github.com/user-attachments/assets/c058311a-fd9c-48df-978a-2dccdc9ba129)

---

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python **3.7+**, with Python **3.13.1** being recommended.
- Pip (Python package manager)

### Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/adpierzchala/ga-sound-generator.git
   cd ga-sound-generator
   ```

2. **Install Dependencies**:
   Install the required Python libraries using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add Audio Samples**:
   Place `.wav` audio samples for different instruments in the `samples` folder. The sample filenames should follow this format:  
   ```
   <NOTE>_<INSTRUMENT>.wav
   ```
   Example:
   - `C_piano.wav`
   - `D_violin.wav`

4. **Run the Application**:
   Start the Flask server:
   ```bash
   python main.py
   ```
   The application will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## How It Works

1. **Genetic Algorithm**:
   - A genetic algorithm is used to compose musical phrases based on user-defined parameters.
   - The fitness function ensures musically coherent phrases by:
     - Penalizing REST Notes:
       - Mild penalty for individual REST notes to discourage excessive pauses.
       - Strong penalty for consecutive REST notes to avoid interruptions in the melody.
     - Rewarding Musical Notes:
       - Adds a score for non-REST notes to encourage rich, melodic phrases.
     - Encouraging Rhythmic Consistency:
       - Rewards consecutive notes with the same duration for maintaining rhythmic stability.
     - Harmonic Intervals Between Consecutive Notes:
       - Reward: Harmonic intervals (e.g., major thirds, perfect fifths) receive a positive score.
       - Penalty: Large melodic leaps (intervals greater than 7 semitones) are discouraged.
     - Harmonic Relationships Between Instruments:
       - Adds a reward when multiple instruments playing the same note exhibit harmonic intervals.
     - Recognizing Repeated Patterns:
       - Rewards phrases that include repeated patterns (e.g., three consecutive notes forming a motif).
     - Balancing Long and Short Notes:
       - Encourages a dynamic phrase by rewarding a balanced mix of long and short notes.
     - Instrument Diversity:
       - Adds a score for using a variety of unique instruments in the phrase.
     - Penalizing Excessive REST Notes:
       - Applies a penalty if REST notes exceed 20% of the total phrase length.
  
2. **Instruments**:
   - Users can choose from predefined instruments via checkboxes.
   - The backend ensures only selected instruments are used in the generated phrase.

3. **Audio Processing**:
   - Audio samples for each note and instrument are combined to create the final composition.
   - The generated audio is normalized and includes fade-in/out effects for smooth transitions.

---

## Project Structure

- `main.py`: The main Flask application with routes for generating, playing, and downloading songs.
- `templates/index.html`: The front-end interface.
- `samples/`: A folder for storing `.wav` audio samples.

---

## Future Enhancements

- Add support for more instruments and note variations.
- Improve the fitness function for more sophisticated compositions.
- Include visualizations of the generated notes (e.g., sheet music or piano roll).

---

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

---

