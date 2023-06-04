from collections import OrderedDict
from copy import deepcopy

from addict import Dict


def parse_param_output(llm_output):
    params_info = OrderedDict()
    params_str = llm_output.split('\nPreconditions:')[0].strip().split('\n\n')[-1]
    for line in params_str.split('\n'):
        if line.strip() == '' or '.' not in line:
            continue
        if not line.split('.')[0].strip().isdigit():
            continue
        try:
            p_info = [e for e in line.split(':')[0].split(' ') if e != '']
            param_name, param_type = p_info[1].strip(), p_info[3].strip()
            params_info[param_name] = param_type
        except Exception:
            print(f'[WARNING] checking param object types - fail to parse: {line}')
            continue
    return params_info


def parse_new_predicates(llm_output):
    new_predicates = list()
    start_idx = llm_output.find('New Predicates:\n')
    if start_idx < 0 or 'No newly defined predicate' in llm_output or 'None' in llm_output:
        pass
    else:
        predicate_output = llm_output[start_idx + len('New Predicates:'):].strip().split('\n\n')[0].strip()
        for p_line in predicate_output.split('\n'):
            if '.' not in p_line or not p_line.split('.')[0].strip().isdigit():
                print(f'[WARNING] unable to parse the line: {p_line}')
                continue
            predicate_name = p_line.split(': ')[0].split('. ')[1].split(' ')[0]
            predicate_name = predicate_name[1:]  # drop the leading (
            predicate_desc = p_line.split(': ')[1].strip()
            # drop the index
            start_idx = p_line.find('. ')
            if start_idx <= 0:
                print(f'[WARNING] invalid predicate output: {p_line}')
            else:
                p_line = p_line[start_idx+2:].strip()
            new_predicates.append({'name': predicate_name, 'desc': predicate_desc, 'raw': p_line})
    return new_predicates


def parse_predicates(all_predicates):
    """
    This function assumes the predicate definitions adhere to PDDL grammar
    """
    all_predicates = deepcopy(all_predicates)
    for i, pred in enumerate(all_predicates):
        if 'params' in pred:
            continue
        pred_def = pred['raw'].split(': ')[0]
        pred_def = pred_def[1:-1].strip()  # discard the parentheses
        split_predicate = pred_def.split(' ')[1:]   # discard the predicate name
        split_predicate = [e for e in split_predicate if e != '']

        pred['params'] = OrderedDict()
        for j, p in enumerate(split_predicate):
            if j % 3 == 0:
                assert '?' in p, f'invalid predicate definition: {pred_def}'
                assert split_predicate[j+1] == '-', f'invalid predicate definition: {pred_def}'
                param_name, param_obj_type = p, split_predicate[j+2]
                pred['params'][param_name] = param_obj_type
    return all_predicates


def read_object_types(hierarchy_info):
    obj_types = set()
    for obj_type in hierarchy_info:
        obj_types.add(obj_type)
        if len(hierarchy_info[obj_type]) > 0:
            obj_types.update(hierarchy_info[obj_type])
    return obj_types


def flatten_pddl_output(pddl_str):
    open_parentheses = 0
    old_count = 0
    flat_str = ''
    pddl_lines = pddl_str.strip().split('\n')
    for line_i, pddl_line in enumerate(pddl_lines):
        pddl_line = pddl_line.strip()
        # process parentheses
        for char in pddl_line:
            if char == '(':
                open_parentheses += 1
            elif char == ')':
                open_parentheses -= 1
        if line_i == 0:
            flat_str += pddl_line + '\n'
        elif line_i == len(pddl_lines) - 1:
            flat_str += pddl_line
        else:
            assert open_parentheses >= 1, f'{open_parentheses}'
            leading_space = ' ' if old_count > 1 else '  '
            if open_parentheses == 1:
                flat_str += leading_space + pddl_line + '\n'
            else:
                flat_str += leading_space + pddl_line
        old_count = open_parentheses
    return flat_str


def parse_full_domain_model(llm_output_dict, action_info):
    def find_leftmost_dot(string):
        for i, char in enumerate(string):
            if char == '.':
                return i
        return 0

    parsed_action_info = Dict()
    for act_name in action_info:
        if act_name in llm_output_dict:
            llm_output = llm_output_dict[act_name]['llm_output']
            try:
                # the first part is parameters
                parsed_action_info[act_name]['parameters'] = list()
                params_str = llm_output.split('\nPreconditions:')[0].strip().split('\n\n')[-1]
                for line in params_str.split('\n'):
                    if line.strip() == '' or '.' not in line:
                        continue
                    if not line.split('.')[0].strip().isdigit():
                        continue
                    leftmost_dot_idx = find_leftmost_dot(line)
                    param_line = line[leftmost_dot_idx + 1:].strip()
                    parsed_action_info[act_name]['parameters'].append(param_line)
                # the second part is preconditions
                parsed_action_info[act_name]['preconditions'] = flatten_pddl_output(llm_output.split('Preconditions:')[1].split('```')[1].strip())
                # the third part is effects
                parsed_action_info[act_name]['effects'] = flatten_pddl_output(llm_output.split('Effects:')[1].split('```')[1].strip())
                # include the act description
                parsed_action_info[act_name]['action_desc'] = llm_output_dict[act_name]['action_desc'] if 'action_desc' in llm_output_dict[act_name] else ''
            except:
                print('[ERROR] errors in parsing pddl output')
                print(llm_output)
    return parsed_action_info






