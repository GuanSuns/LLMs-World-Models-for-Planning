def read_predicates(predicate_file, contain_idx=False):
    def find_leftmost_dot(string):
        for i, char in enumerate(string):
            if char == '.':
                return i
        return None

    predicate_list = list()

    with open(predicate_file) as f:
        predicates = f.read().strip()
    if predicates == '':
        return predicate_list

    for pred_item in predicates.split('\n'):
        if contain_idx:
            start_idx = find_leftmost_dot(pred_item)
            assert start_idx is not None, f'{pred_item}'
            pred_line = pred_item[start_idx + 1:].strip()
        else:
            pred_line = pred_item.strip()
        predicate_name = pred_line.split(': ')[0].split(' ')[0]
        predicate_name = predicate_name[1:]  # drop the leading (
        predicate_desc = pred_line.split(': ')[1].strip()
        predicate_list.append({'name': predicate_name, 'desc': predicate_desc, 'raw': pred_line})

    from utils.pddl_output_utils import parse_predicates
    predicate_list = parse_predicates(predicate_list)
    return predicate_list
