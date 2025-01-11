import os
import numpy as np
import random
from flask import Flask, request, render_template, send_file, jsonify
from scipy.io.wavfile import write, read
from scipy.signal import resample

# Constants and parameters
NOTES = ["C", "D", "E", "F", "G", "A", "B"] # Musical notes
SAMPLE_RATE = 48000 # Audio sample rate in Hz
ALLOWED_INSTRUMENTS = ["piano", "violin", "flute"] # Default instruments
SAMPLE_DIR = SAMPLE_DIR = "C:/Users/Uzytkownik/Downloads/samples" # Path to sample files

# Flask app initialization
app = Flask(__name__)


# Normalize audio
def normalize_audio(data):
    """Normalize audio to the range of int16."""
    if np.max(np.abs(data)) == 0: # Avoid division by zero
        return data
    return (data / np.max(np.abs(data)) * 32767).astype(np.int16) 


# Apply fade-in and fade-out 
def apply_fade(data, fade_duration=0.01):
    """Apply fade-in and fade-out to an audio sample."""
    fade_samples = int(fade_duration * SAMPLE_RATE)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    data[:fade_samples] = data[:fade_samples] * fade_in
    data[-fade_samples:] = data[-fade_samples:] * fade_out
    return data

# Apply fade-out 
def apply_fade_out(song, fade_duration=0.05):
    """Apply a fade-out to the entire song."""
    fade_samples = int(fade_duration * SAMPLE_RATE)
    if len(song) < fade_samples:
        fade_samples = len(song) # Adjust for very short songs
    fade_out = np.linspace(1, 0, fade_samples)
    song[-fade_samples:] = (song[-fade_samples:] * fade_out).astype(np.int16)
    return song


# Load a sample
def load_sample(instrument, note, tempo):
    """Load the appropriate audio sample for a given note and instrument."""
    if note == "REST":
        # Generate silence for a rest
        return np.zeros(int(SAMPLE_RATE * 60 / tempo * 0.5), dtype=np.int16)
    file_path = os.path.join(SAMPLE_DIR, f"{note}_{instrument}.wav")

    try:
        rate, data = read(file_path)
        if rate != SAMPLE_RATE:
            # Resample to match the desired sample rate
            num_samples = int(len(data) * SAMPLE_RATE / rate)
            # Convert stereo to mono by averaging channels
            data = resample(data, num_samples).astype(np.int16)
        if len(data.shape) == 2:
            data = data.mean(axis=1).astype(np.int16)
        data = normalize_audio(data) # Normalize the audio
        data = apply_fade(data) # Apply fade effect
        return data
    except FileNotFoundError:
        print(f"Sample file {file_path} not found.")
        # Return silence if the sample file is missing
        return np.zeros(int(SAMPLE_RATE * 60 / tempo * 0.5), dtype=np.int16)


# Play a phrase
def play_phrase_with_samples(phrase, tempo):
    """Generate a full song from a phrase using the provided samples."""
    song = np.array([], dtype=np.int16)
    for note in phrase:
        note_name = note["note"]
        duration = 60 / tempo * note["duration"]
        expected_length = int(SAMPLE_RATE * duration)

        # Combine sounds from multiple instruments
        combined_sample = np.zeros(expected_length, dtype=np.float32)
        for instrument in note["instruments"]:
            sample = load_sample(instrument, note_name, tempo)
            if len(sample) > expected_length:
                sample = sample[:expected_length]
            else:
                sample = np.pad(sample, (0, expected_length - len(sample)))
            combined_sample += sample

        # Normalize the combined sample
        combined_sample = normalize_audio(combined_sample)

        # Concatenate to the final song
        song = np.concatenate((song, combined_sample))

    # Apply fade-out effect to the full song 
    song = apply_fade_out(song, fade_duration=0.5)
    return normalize_audio(song) # Final normalization


# Generate an initial population of musical phrases
def generate_initial_population_with_notes(pop_size, num_notes, instruments):
    """Create an initial population of musical phrases with a fixed number of notes."""
    population = []
    for _ in range(pop_size):
        phrase = []
        for _ in range(num_notes):
            note_duration = random.choice([0.25, 0.5, 1])  # Random duration
            note = random.choice(NOTES + ["REST"])  # Random note or REST
            
            instruments_for_note = [] if note == "REST" else random.sample(instruments, len(instruments))
            instruments_for_note = list(set(instruments_for_note))

            phrase.append({
                "note": note,
                "duration": note_duration,
                "instruments": instruments_for_note,
            })
        population.append(phrase)
    return population

HARMONIC_INTERVALS = [0, 2, 4, 5, 7, 9, 11]

# Fitness function 
def fitness_function(phrase):
    """Evaluate the fitness of a musical phrase."""
    score = 0
    patterns = set()
    instruments_used = set()
    long_notes = 0
    short_notes = 0
    rest_count = 0

    for i, note in enumerate(phrase):
        # Handle REST notes
        if note["note"] == "REST":
            rest_count += 1

            # Penalize consecutive REST notes
            if i > 0 and phrase[i - 1]["note"] == "REST":
                score -= 2  # Strong penalty for consecutive rests
            else:
                score -= 0.5  # Mild penalty for individual REST notes
        else:
            # Reward musical notes
            score += 1

        # Reward rhythmic consistency
        if i > 0 and note["duration"] == phrase[i - 1]["duration"]:
            score += 0.5

        # Reward harmonic relationships between consecutive notes
        if i > 0 and note["note"] != "REST" and phrase[i - 1]["note"] != "REST":
            interval = abs(NOTES.index(note["note"]) - NOTES.index(phrase[i - 1]["note"]))
            if interval in HARMONIC_INTERVALS:
                score += 1  # Reward for harmonic intervals
            elif interval > 7:
                score -= 1  # Penalize large leaps

        # Reward harmonic relationships between instruments playing the same note
        if note["note"] != "REST":
            note_instruments = note.get("instruments", ["piano"])
            if len(note_instruments) > 1:
                for j in range(len(note_instruments)):
                    for k in range(j + 1, len(note_instruments)):
                        interval = abs(NOTES.index(note["note"]))
                        if interval in HARMONIC_INTERVALS:
                            score += 0.5  # Smaller reward for harmonic intervals between instruments

            instruments_used.update(note_instruments)

        # Reward repeated patterns
        if i >= 2:
            pattern = (phrase[i - 2]["note"], phrase[i - 1]["note"], note["note"])
            if pattern in patterns:
                score += 1  # Reward for repeating patterns
            patterns.add(pattern)

        # Track long and short notes
        if note["duration"] > 0.5:
            long_notes += 1
        elif note["duration"] <= 0.5:
            short_notes += 1

    # Reward balanced use of long and short notes
    balance = min(long_notes, short_notes)
    score += balance * 0.5

    # Add score based on the number of unique instruments used across the phrase
    score += len(instruments_used) * 0.5

    # Penalize too many REST notes overall
    if rest_count > len(phrase) * 0.2:
        score -= (rest_count - len(phrase) * 0.2) * 0.5

    return score


# Genetic algorithm
def genetic_algorithm(population, generations, mutation_rate, allowed_instruments):
    """Optimize a population of phrases using a genetic algorithm."""
    for _ in range(generations):
        # Calculate fitness for the current population
        fitness_scores = [fitness_function(ind) for ind in population]

        # Sort population based on fitness (higher is better)
        sorted_population = [
            x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)
        ]

        # Select parents (top half of the population)
        parents = sorted_population[: len(population) // 2]
        new_population = parents[:]

        # Create offspring using crossover
        while len(new_population) < len(population):
            parent1, parent2 = random.sample(parents, 2)
            crossover_point = random.randint(1, len(parent1) - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            new_population.append(child)

        # Apply mutation
        for individual in new_population:
            if random.random() < mutation_rate:
                mutate_point = random.randint(0, len(individual) - 1)
                individual[mutate_point] = {
                    "note": random.choice(NOTES + ["REST"]),
                    "duration": random.choice([0.25, 0.5, 1]),
                    "instruments": [] if individual[mutate_point]["note"] == "REST" else random.sample(allowed_instruments, len(allowed_instruments)),
                }

        # Update the population
        population = new_population

    # Return the best individual (highest fitness)
    return sorted_population[0]

# Flask routes
@app.route("/")
def index():
    return render_template("index.html") # Render the main interface

# Route to handle the generation of a new song
@app.route("/generate/", methods=["POST"])
def generate():
    """
    Handle POST requests to generate a song using genetic algorithm parameters.
    Accepts user-defined tempo, instruments, phrase length, number of generations, and mutation rate.
    """
    global generated_song, generated_notes

    try:
        # Extract parameters from the form submission
        tempo = int(request.form["tempo"])

        # Extract selected instruments from the form
        instruments = request.form.getlist("instruments")

        # Filter the instruments to ensure they are valid and in the allowed list
        instruments = [inst.lower() for inst in instruments if inst.lower() in ALLOWED_INSTRUMENTS]

        phrase_length = int(request.form["phrase_length"])
        num_generations = int(request.form["num_generations"])
        mutation_rate = float(request.form["mutation_rate"]) 

        # Ensure at least one valid instrument is selected
        if not instruments:
            return jsonify({"error": "No valid instruments entered."}), 400

        # Generate initial population of phrases and evolve using the genetic algorithm
        population = generate_initial_population_with_notes(10, phrase_length, instruments)
        best_phrase = genetic_algorithm(
            population, 
            generations=num_generations, 
            mutation_rate=mutation_rate, 
            allowed_instruments=instruments)
        
        # Store the generated notes and convert them into audio
        generated_notes = best_phrase
        generated_song = play_phrase_with_samples(best_phrase, tempo)

        # Return success message
        return jsonify({"message": "Song generated successfully!"})
    except Exception as e:

        # Handle and return errors
        return jsonify({"error": str(e)}), 400

# Route to handle playing the generated song
@app.route("/play/", methods=["GET"])
def play():
    """
    Serve the generated song as a WAV file to the user.
    """

    global generated_song

    # Check if a song has been generated
    if generated_song is None:
        return jsonify({"error": "No song generated yet."}), 400

    # Define the file path for saving the generated song
    output_file = os.path.join(os.getcwd(), "generated_song.wav")
    
    # Write the generated song to a WAV file
    write(output_file, SAMPLE_RATE, generated_song)

    # Ensure the file exists before sending it
    if not os.path.exists(output_file):
        return jsonify({"error": f"File not found: {output_file}"}), 404
    
    # Send the file as a response
    return send_file(output_file, as_attachment=False, mimetype="audio/wav")

# Route to fetch the generated notes as JSON
@app.route("/notes/", methods=["GET"])
def notes():
    """
    Provide the generated musical notes as a JSON response.
    """
    global generated_notes

    # Check if notes have been generated
    if generated_notes is None:
        return jsonify({"error": "No notes generated yet."}), 400

    # Format the notes for the response
    notes = [
        f"{note['note']} - {', '.join(note['instruments'])}"
        for note in generated_notes
    ]

    # Return the notes as a JSON response
    return jsonify({"notes": notes})

# Route to allow downloading the generated song
@app.route("/download/", methods=["GET"])
def download():
    """
    Allow the user to download the generated song as a WAV file.
    """
    global generated_song

    # Check if a song has been generated
    if generated_song is None:
        return jsonify({"error": "No song generated yet."}), 400

    # Define the file path for the generated song
    output_file = os.path.join(os.getcwd(), "generated_song.wav")
    # Write the song to a file
    write(output_file, SAMPLE_RATE, generated_song)

    # Send the file as a downloadable response
    return send_file(output_file, as_attachment=True, mimetype="audio/wav")

# Main entry point for running the Flask app
if __name__ == "__main__":
    app.run(debug=True)
