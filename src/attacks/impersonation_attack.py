import copy, random

import numpy as np

def impersonation_attack(dataset, subject_id):
    positive_sequences = [*filter(lambda mouse_event_sequence: mouse_event_sequence.subject_id == subject_id, dataset)]
    negative_sequences = [*filter(lambda mouse_event_sequence: mouse_event_sequence.subject_id != subject_id, dataset)]
    impersonation_sequences = []

    for _ in range(len(positive_sequences)):
        impersonation_sequence = copy.deepcopy(random.choice(negative_sequences))
        impersonation_sequence.subject_id = subject_id
        for mouse_event in impersonation_sequence.mouse_event_sequence:
            mouse_event.x += int(np.random.normal(0, 15))
            mouse_event.y += int(np.random.normal(0, 15))
        impersonation_sequences.append(impersonation_sequence)

    dataset.extend(impersonation_sequences)
    return dataset
