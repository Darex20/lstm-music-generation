# Required Libraries
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

def load_midi_files(main_directories):
    midi_files = []
    for directory in main_directories:
        for root, dirs, files in os.walk(directory):
            # Skip the "versions" subdirectories
            if 'versions' in root:
                continue
            for file in files:
                if file.endswith('.mid') or file.endswith('.midi'):
                    midi_files.append(os.path.join(root, file))
    return midi_files

# Function to Extract Notes from MIDI Files
def get_notes_from_midi(midi_files):
    notes = []
    for file in midi_files:
        midi = converter.parse(file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        if parts: 
            notes_to_parse = parts.parts[0].recurse()
        else: 
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

# Prepare Sequences for LSTM
def prepare_sequences(notes, n_vocab, sequence_length=100):
    note_to_int = dict((note, number) for number, note in enumerate(set(notes)))
    network_input = []
    network_output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output)
    return (network_input, network_output, note_to_int)

# Build LSTM Model
def create_model(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), recurrent_dropout=0.3, return_sequences=True))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

# Generate Notes from the Model
def generate_notes(model, network_input, note_to_int, int_to_note, temperature=0.5):
    start = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start]
    prediction_output = []

    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(len(note_to_int))
        prediction = model.predict(prediction_input, verbose=0)
        index = sample(prediction[0], temperature)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

# Convert Predictions to MIDI File
def create_midi(prediction_output, output_file='test_output.mid'):
    offset = 0
    output_notes = []
    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)

# Function to Introduce Randomness in Predictions
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds += 1e-10  # Add a small constant to avoid log(0)
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Main Execution
if __name__ == '__main__':
    # Path to MIDI files
    data_directory = './data'
    pop909_directory = './POP909-Dataset/POP909'

    midi_files = load_midi_files([data_directory])
    # midi_files = [os.path.join(path_to_midi, file) for file in os.listdir(path_to_midi) if file.endswith('.mid')]
    
    # Extract notes from MIDI files
    notes = get_notes_from_midi(midi_files)
    n_vocab = len(set(notes))
    
    # Prepare sequences
    network_input, network_output, note_to_int = prepare_sequences(notes, n_vocab)
    int_to_note = dict((number, note) for number, note in enumerate(set(notes)))
    
    # Create the model
    model = create_model(network_input, n_vocab)
    
    # Train the model
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(network_input, network_output, epochs=10, batch_size=128, callbacks=callbacks_list)
    
    # Generate music
    prediction_output = generate_notes(model, network_input, note_to_int, int_to_note)
    
    # Create MIDI file from generated notes
    create_midi(prediction_output)
