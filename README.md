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
# Documentation

## 1. Software Description

### a. Short Name:
**GA Sound Generator**

### b. Full Name:
**Genetic Algorithm-Based Musical Phrase Generator**

### c. Brief Description with Objectives:
The "GA Sound Generator" project allows users to generate musical phrases using genetic algorithms. The goal of the project is to create a flexible tool that lets users manipulate parameters such as tempo, instruments, and phrase length to produce unique melodies in .wav format. The application provides an interactive web interface for easy user interaction with the system. The generator can be used in music education, music prototyping, and creative sound applications.

## 2. Copyright

### a. Authors:
- Kamila Mańska
- Adrianna Pierzchała

### b. Licensing Terms:
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## 3. Requirements Specification

| ID  | NAME                  | DESCRIPTION                                                           | PRIORITY | CATEGORY    |
| --- | ---------------------- | --------------------------------------------------------------------- | -------- | ----------- |
| R1  | Parameterization       | The user can select instruments (piano, violin, flute), enter the tempo and phrase length. | 1        | Functional  |
| R2  | GA Parameter Adjustment| Allows setting the number of generations and mutation rate.          | 3        | Functional  |
| R3  | Phrase Generation      | The system generates a musical phrase based on user-provided parameters. | 1        | Functional  |
| R4  | Playing a musical phrase| The system allows you to listen to the generated phrase using the built-in player. | 1        | Functional  |
| R5  | Note Display           | Generated notes are displayed in the interface.                       | 3        | Functional  |
| R6  | File Download          | The generated .wav file can be downloaded by the user.               | 2        | Functional  |
| R7  | Error handling         | The system displays error messages.                                  | 2        | Functional  |
| R8  | Scalability            | The system supports musical phrases of possibly different sizes.     | 3        | Non-functional |
| R9  | Compatibility          | The application runs on Windows, Linux, iOS with appropriate libraries. | 2        | Non-functional |
| R10 | Intuitive Interface    | The interface should be clear and easy to use.                        | 2        | Non-functional |

## 4. System/Software Architecture

### a. Development Architecture
Tools and technologies used during the development of the software:

- **Python**: Main programming language (Version: 3.13.1)
- **HTML**: Web structure (Version: 5.0)
- **Bootstrap**: Interface Styling (Version: 5.1.3)
- **JavaScript**: Frontend web development (ES6)
- **Visual Studio Code**: Code Editor (Version: 1.96.2)

Libraries:
- **Flask**: Web framework (Version: 2.3.2)
- **NumPy**: Data processing (Version: 1.24.3)
- **SciPy**: Signal processing (Version: 1.11.2)

### b. Deployment Architecture
Tools and technologies required to run the software in the target environment:

- **Python Interpreter**: Backend execution (Version: 3.7+)
- **Web browser**: User interface (Any)

Libraries:

- **Flask**: Web framework (Version: 2.3.2)
- **NumPy**: Data processing (Version: 1.24.3)
- **SciPy**: Signal processing (Version: 1.11.2)

## 5. Tests

### a. Test Scenarios:

| ID  | Test description                                                                 |
| --- | --------------------------------------------------------------------------------- |
| T1  | The user can enter the tempo (bpm), select specific instruments, and the length of the musical phrase. |
| T2  | The user can modify genetic algorithm parameters (generations, mutation rate).  |
| T3  | The user can generate a musical phrase by clicking the "Generate" button.       |
| T4  | The user can play the generated phrase in the browser.                          |
| T5  | The system displays or hides notes corresponding to a phrase.                    |
| T6  | The user can download the generated .wav file.                                  |
| T7  | The system handles errors when required inputs are missing (e.g., no instruments). |

### b. Test Execution Report:

| ID  | Results  | Notes                               |
| --- | -------- | ----------------------------------- |
| T1  | Success  | User input works as expected.       |
| T2  | Success  | Parameters adjust correctly.        |
| T3  | Success  | Phrase generated successfully.      |
| T4  | Success  | Audio playback works correctly.    |
| T5  | Success  | Notes display toggles properly.     |
| T6  | Success  | File downloaded without issues.    |
| T7  | Success  | Error messages displayed.          |

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

1. **Download and Extract the Project Files**:
   Once the ZIP file is downloaded, extract its contents to a folder on your computer.

3. **Install Dependencies**:
   Install the required Python libraries which are in the `requirements.txt` file.

4. **Add Audio Samples**:
   Modify the path to sample files in the variable `SAMPLE_DIR` in `main.py` file.

5. **Run the Application**:
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
     - Recognizing Repeated Patterns:
       - Rewards phrases that include repeated patterns (e.g., three consecutive notes forming a motif).
     - Balancing Long and Short Notes:
       - Encourages a dynamic phrase by rewarding a balanced mix of long and short notes.
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

