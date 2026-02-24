import copy

from ..utils.variance_model import VAR

def impersonation_attack(dataset, subject_id):
    mouse_event_sequences = [*filter(lambda mouse_event_sequence: mouse_event_sequence.subject_id == subject_id, dataset)]
    variance_model = VAR().fit(mouse_event_sequences)
    impersonation_sequences = []

    for mouse_event_sequence in dataset:
        if mouse_event_sequence.subject_id == subject_id:
            impersonation_sequences.append(mouse_event_sequence)
        else:
            impersonation_sequence = copy.deepcopy(mouse_event_sequence)
            deltas_len = len(impersonation_sequence.mouse_event_sequence) - 1
            deltas = variance_model.sample(deltas_len)

            mouse_events = impersonation_sequence.mouse_event_sequence
            for i in range(1, len(mouse_events)):
                mouse_events[i].x = int(round(mouse_events[i - 1].x - deltas[i - 1, 0]))
                mouse_events[i].y = int(round(mouse_events[i - 1].y - deltas[i - 1, 1]))
            impersonation_sequences.append(impersonation_sequence)

    return impersonation_sequences
