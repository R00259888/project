import copy, random

import numpy as np

def data_augmentation_defence(dataset, subject_id):
    subject_sequences = [*filter(lambda mouse_event_sequence: mouse_event_sequence.subject_id == subject_id, dataset)]
    augmented_mouse_event_sequences = []

    for mouse_event_sequence in subject_sequences:
        augmented_mouse_event_sequence = copy.deepcopy(mouse_event_sequence)
        mouse_event_sequence_length = len(augmented_mouse_event_sequence.mouse_event_sequence)
        mutation_count = random.randint(max(1, int(mouse_event_sequence_length * 0.2)), max(1, int(mouse_event_sequence_length * 0.5)))

        for index in random.sample(range(mouse_event_sequence_length), min(mutation_count, mouse_event_sequence_length)):
            augmented_mouse_event_sequence.mouse_event_sequence[index].x += int(np.random.normal(0, 5))
            augmented_mouse_event_sequence.mouse_event_sequence[index].y += int(np.random.normal(0, 5))

        augmented_mouse_event_sequences.append(augmented_mouse_event_sequence)

    dataset.extend(augmented_mouse_event_sequences)
    return dataset
