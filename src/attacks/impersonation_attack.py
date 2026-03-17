from ..utils.var_model import VAR

def impersonation_attack(dataset, subject_id):
    mouse_event_sequences = [*filter(lambda mouse_event_sequence: mouse_event_sequence.subject_id == subject_id, dataset)]
    var_model = VAR().fit(mouse_event_sequences)
    impersonation_sequences = []

    for mouse_event_sequence in dataset:
        if mouse_event_sequence.subject_id == subject_id:
            impersonation_sequences.append(mouse_event_sequence)
        else:
            deltas = var_model.sample(len(mouse_event_sequence) - 1)
            impersonation_sequences.append(mouse_event_sequence.replace_mouse_event_sequence(deltas))

    return impersonation_sequences
