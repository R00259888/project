import copy, random

import numpy as np

from ..dataset.mouse_dynamics import MouseEvent, MouseEventSequence
from ..utils.variance_model import VAR

def __generate_negative_sequence(model, length):
    deltas_len = length - 1
    deltas = model.sample(deltas_len)

    mouse_event = [MouseEvent(timestamp=0, x=0, y=0, button_pressed=0, subject_id=-1)]
    for i in range(deltas_len):
        mouse_event.append(MouseEvent(
            timestamp=i + 1,
            x=int(round(mouse_event[-1].x - deltas[i, 0])),
            y=int(round(mouse_event[-1].y - deltas[i, 1])),
            button_pressed=0,
            subject_id=-1,
        ))
    return MouseEventSequence(mouse_event)

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

    variance_model = VAR().fit(subject_sequences)
    for _ in range(len(subject_sequences)):
        negative_sequence = __generate_negative_sequence(variance_model, MouseEventSequence.CHUNK_SIZE)
        augmented_mouse_event_sequences.append(negative_sequence)

    dataset.extend(augmented_mouse_event_sequences)
    return dataset
