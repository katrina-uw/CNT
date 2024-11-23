import numpy as np

def get_anomaly_sequences(values):
    splits = np.where(values[1:] != values[:-1])[0] + 1
    if values[0] == 1:
        splits = np.insert(splits, 0, 0)

    a_seqs = []
    for i in range(0, len(splits) - 1, 2):
        a_seqs.append([splits[i], splits[i + 1] - 1])

    if len(splits) % 2 == 1:
        a_seqs.append([splits[-1], len(values) - 1])

    return a_seqs


def create_shapes(ranges, sequence_type, _min, _max, plot_values, xref=None, yref=None):
    if _max is None:
        _max = max(plot_values["errors"])

    if sequence_type is None:
        color = "blue"
    else:
        if sequence_type == "true":
            color = "red"
        elif sequence_type == "predicted":
            color = "blue"
        elif sequence_type == "individual_true":
            color = "green"
    shapes = []

    for r in ranges:
        w = 5
        x0 = r[0] - w
        x1 = r[1] + w
        shape = {
            "type": "rect",
            "x0": x0,
            "y0": _min,
            "x1": x1,
            "y1": _max,
            "fillcolor": color,
            "opacity": 0.08,
            "line": {
                "width": 0,
            },
        }
        if xref is not None:
            shape["xref"] = xref
            shape["yref"] = yref

        shapes.append(shape)

    return shapes



