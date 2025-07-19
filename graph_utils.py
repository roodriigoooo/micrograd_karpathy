def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    modified to handle both scalars and tensors
    """
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    nodes, edges = trace(root)
    for node in nodes:
        uid = str(id(node))
        if node.data.size == 1:
            label_text = "{ %s | data %.4f | grad %.4f }" % (node.label, node.data, node.grad)
        else:
            label_text = "{ %s | data %s | grad mean: %.4f}" % (node.label or '', str(node.data.shape), np.mean(node.grad))

        dot.node(name=uid, label=label_text, shape='record')

        if node._op:
            dot.node(name=uid + node._op, label=node._op)
            dot.edge(uid + node._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
