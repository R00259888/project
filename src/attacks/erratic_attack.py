import copy, random

import numpy as np

def erratic_attack(dataset, subject_id):
    subject_mouse_event_sequences = [*filter(lambda mouse_event_sequence: mouse_event_sequence.subject_id == subject_id, dataset)]
    erratic_mouse_event_sequences = []

    for _ in range(1):
        erratic_mouse_event_sequence = copy.deepcopy(random.choice(subject_mouse_event_sequences))
        for mouse_event in erratic_mouse_event_sequence.mouse_event_sequence:
            mouse_event.x += int(np.random.normal(0, 15))
            mouse_event.y += int(np.random.normal(0, 15))
        erratic_mouse_event_sequences.append(erratic_mouse_event_sequence)

    dataset.extend(erratic_mouse_event_sequences)
    return dataset
