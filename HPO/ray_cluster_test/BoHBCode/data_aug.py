import nltk
import fasttext.util
import language_tool_python
import random
import threading

# Download NLTK resources if not already downloaded
nltk.download('punkt')

# Load FastText word vectors
fasttext.util.download_model('de', if_exists='ignore')  # Download the German model
ft = fasttext.load_model('cc.de.300.bin')

# Initialize LanguageTool for German
tool = language_tool_python.LanguageTool('de')

# Function to check grammatical correctness
def is_grammatically_correct(sentence):
    matches = tool.check(sentence)
    return len(matches) == 0

# Function to replace words in a sentence with similar ones using FastText embeddings
def replace_with_similar_words(sentence, progress_event):
    words = nltk.word_tokenize(sentence)
    new_sentence = []
    for word in words:
        similar_words = ft.get_nearest_neighbors(word, k=5)  # Get 5 nearest neighbors
        if similar_words:
            new_word = random.choice(similar_words)[1]
            new_sentence.append(new_word)
        else:
            new_sentence.append(word)
        if progress_event.is_set():
            break  # Exit loop if progress event is set
    new_sentence = ' '.join(new_sentence)
    # Check grammatical correctness
    if is_grammatically_correct(new_sentence):
        return new_sentence
    else:
        return None

# Function to generate new sentences with progress indication
def generate_new_sentences(original_sentence):
    progress_event = threading.Event()
    progress_event.clear()  # Reset progress event
    for _ in range(3):
        new_sentence = None
        while new_sentence is None or new_sentence == original_sentence:
            new_sentence = replace_with_similar_words(original_sentence, progress_event)
        print("New sentence:", new_sentence)
    progress_event.set()  # Set progress event after completion

# Original array of German sentences
german_sentences = [
    "Die Sonne scheint hell am Himmel.",
    "Ich liebe es, im Wald spazieren zu gehen.",
    "Guten Morgen! Wie geht es Ihnen heute?"
]

# Generate and print new sentences for each original sentence
for original_sentence in german_sentences:
    print("Original sentence:", original_sentence)
    generate_new_sentences(original_sentence)
    print()  # Empty line for clarity between each original sentence and its new sentences
