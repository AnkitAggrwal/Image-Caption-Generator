# utils/caption_generator.py

import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences

def beam_search_caption(model, image_features, word_to_index, index_to_word, max_length=38, beam_width=3):
    """
    Generate a caption using beam search.
    """
    start = [word_to_index['<start>']]
    sequences = [[start, 0.0]]

    while len(sequences[0][0]) < max_length:
        all_candidates = []
        for seq, score in sequences:
            padded_seq = pad_sequences([seq], maxlen=max_length)
            yhat = model.predict([image_features, padded_seq], verbose=0)
            top_ids = np.argsort(yhat[0])[-beam_width:]

            for word_id in top_ids:
                new_seq = seq + [word_id]
                new_score = score + np.log(yhat[0][word_id] + 1e-9)
                all_candidates.append([new_seq, new_score])

        sequences = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)[:beam_width]
        if all(index_to_word.get(seq[-1], '') == '<end>' for seq, _ in sequences):
            break

    final_seq = sequences[0][0]
    final_words = [index_to_word.get(i, '') for i in final_seq]
    return ' '.join([w for w in final_words if w not in ['<start>', '<end>', '<pad>']])
