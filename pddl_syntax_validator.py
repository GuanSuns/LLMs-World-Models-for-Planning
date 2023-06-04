from copy import deepcopy

from utils.pddl_output_utils import parse_param_output, parse_new_predicates, parse_predicates, read_object_types


class PDDL_Syntax_Validator:
    def __init__(self, obj_hierarchy_info,
                 error_types=None, messed_output_len=20, unsupported_keywords=None):
        default_error_types = ['messed_output_len',
                               'unsupported_keywords',
                               'invalid_param_types',
                               'invalid_predicate_name',
                               'invalid_predicate_format',
                               'invalid_predicate_usage']
        default_unsupported = ['forall', 'when', 'exists', 'implies']
        self.error_types = default_error_types if error_types is None else error_types
        self.unsupported_keywords = unsupported_keywords if unsupported_keywords else default_unsupported
        self.messed_output_len = messed_output_len
        self.obj_types = read_object_types(obj_hierarchy_info)
        self.obj_hierarchy_info = obj_hierarchy_info

    def perform_validation(self, llm_output, **kwargs):
        for error_type in self.error_types:
            if error_type == 'messed_output_len':
                validation_info = self.check_messed_output(llm_output, **kwargs)
            elif error_type == 'unsupported_keywords':
                validation_info = self.check_unsupported_keywords(llm_output, **kwargs)
            elif error_type == 'invalid_param_types':
                validation_info = self.check_param_types(llm_output, **kwargs)
            elif error_type == 'invalid_predicate_name':
                validation_info = self.check_predicate_names(llm_output, **kwargs)
            elif error_type == 'invalid_predicate_format':
                validation_info = self.check_predicate_format(llm_output, **kwargs)
            elif error_type == 'invalid_predicate_usage':
                validation_info = self.check_predicate_usage(llm_output, **kwargs)
            else:
                raise NotImplementedError
            if not validation_info[0]:
                return validation_info
        return True, 'all_validation_pass', None, None

    def check_unsupported_keywords(self, llm_output, **kwargs):
        """
        A simple function to check whether the pddl model uses unsupported logic keywords
        """
        for keyword in self.unsupported_keywords:
            if f'({keyword} ' in llm_output:
                feedback_message = f'The precondition or effect contain the keyword `{keyword}` that is not supported in a standard STRIPS style model. Please express the same logic in a simplified way. You can come up with new predicates if needed (but note that you should use existing predicates as much as possible).\n\nParameters:'
                return False, 'has_unsupported_keywords', keyword, feedback_message
        return True, 'has_unsupported_keywords', None, None

    def check_messed_output(self, llm_output, **kwargs):
        """
        Though this happens extremely rarely, the LLM (even GPT-4) might generate messed-up outputs (basically
            listing a large number of predicates in preconditions and effects)
        """
        assert '\nPreconditions:' in llm_output, llm_output
        precond_str = llm_output.split('\nPreconditions:')[1].split('```\n')[1].strip()
        if len(precond_str.split('\n')) > self.messed_output_len:
            feedback_message = f'You seem to have generated an action model with an unusually long list of preconditions. Please include only the relevant preconditions/effects and keep the action model concise.\n\nParameters:'
            return False, 'messed_output_feedback', None, feedback_message

        return True, 'messed_output_feedback', None, None

    def check_param_types(self, llm_output, **kwargs):
        params_info = parse_param_output(llm_output)
        for param_name in params_info:
            param_type = params_info[param_name]
            if param_type not in self.obj_types:
                feedback_message = f'There is an invalid object type `{param_type}` for the parameter {param_name}. Please revise the PDDL model to fix this error.\n\nParameters:'
                return False, 'invalid_object_type', param_name, feedback_message
        return True, 'invalid_object_type', None, None

    def check_predicate_names(self, llm_output, **kwargs):
        curr_predicates = kwargs['curr_predicates']
        curr_pred_names = {pred['name'].lower(): pred for pred in curr_predicates}
        new_predicates = parse_new_predicates(llm_output)

        # check name clash with obj types
        invalid_preds = list()
        for new_pred in new_predicates:
            curr_obj_types = {t.lower() for t in self.obj_types}
            if new_pred['name'].lower() in curr_obj_types:
                invalid_preds.append(new_pred['name'])
        if len(invalid_preds) > 0:
            feedback_message = f'The following predicate(s) have the same name(s) as existing object types:'
            for pred_i, pred_name in enumerate(list(invalid_preds)):
                feedback_message += f'\n{pred_i + 1}. {pred_name}'
            feedback_message += '\nPlease rename these predicates.\n\nParameters:'
            return False, 'invalid_predicate_names', None, feedback_message

        # check name clash with existing predicates
        duplicated_predicates = list()
        for new_pred in new_predicates:
            if new_pred['name'].lower() in curr_pred_names:
                duplicated_predicates.append((new_pred['raw'], curr_pred_names[new_pred['name'].lower()]['raw']))
        if len(duplicated_predicates) > 0:
            feedback_message = f'The following predicate(s) have the same name(s) as existing predicate(s):'
            for pred_i, duplicated_pred_info in enumerate(duplicated_predicates):
                new_pred_full, existing_pred_full = duplicated_pred_info
                feedback_message += f'\n{pred_i + 1}. {new_pred_full.replace(":", ",")}; existing predicate with the same name: {existing_pred_full.replace(":", ",")}'
            feedback_message += '\n\nYou should reuse existing predicates whenever possible. If you are reusing existing predicate(s), you shouldn\'t list them under \'New Predicates\'. If existing predicates are not enough and you are devising new predicate(s), please use names that are different from existing ones.'
            feedback_message += '\n\nPlease revise the PDDL model to fix this error.\n\n'
            feedback_message += 'Parameters:'
            return False, 'invalid_predicate_names', None, feedback_message

        return True, 'invalid_predicate_names', None, None

    def check_predicate_format(self, llm_output, **kwargs):
        """
        Though this happens rarely, the LLM (even GPT-4) might forget to define the object type of some parameters in new predicates
        """
        new_predicates = parse_new_predicates(llm_output)
        for new_pred in new_predicates:
            new_pred_def = new_pred['raw'].split(': ')[0]
            new_pred_def = new_pred_def[1:-1].strip()   # discard the parentheses
            split_predicate = new_pred_def.split(' ')[1:]   # discard the predicate name
            split_predicate = [e for e in split_predicate if e != '']

            for i, p in enumerate(split_predicate):
                if i % 3 == 0:
                    if '?' not in p:
                        feedback_message = f'There are syntax errors in the definition of the new predicate {new_pred_def}. Please revise its definition and output the entire PDDL action model again. Note that you need to strictly follow the syntax of PDDL.\n\nParameters:'
                        return False, 'invalid_predicate_format', None, feedback_message
                    else:
                        if i + 1 >= len(split_predicate) or split_predicate[i+1] != '-':
                            feedback_message = f'There are syntax errors in the definition of the new predicate {new_pred_def}. Please revise its definition and output the entire PDDL action model again. Note that you need to define the object type of each parameter and strictly follow the syntax of PDDL.\n\nParameters:'
                            return False, 'invalid_predicate_format', None, feedback_message
                        if i + 2 >= len(split_predicate):
                            feedback_message = f'There are syntax errors in the definition of the new predicate {new_pred_def}. Please revise its definition and output the entire PDDL action model again. Note that you need to define the object type of each parameter and strictly follow the syntax of PDDL.\n\nParameters:'
                            return False, 'invalid_predicate_format', None, feedback_message
                        param_obj_type = split_predicate[i+2]
                        if param_obj_type not in self.obj_types:
                            feedback_message = f'There is an invalid object type `{param_obj_type}` for the parameter {p} in the definition of the new predicate {new_pred_def}. Please revise its definition and output the entire PDDL action model again.\n\nParameters:'
                            return False, 'invalid_predicate_format', None, feedback_message
        return True, 'invalid_predicate_format', None, None

    def _is_valid_type(self, target_type, curr_type):
        if target_type == curr_type:
            return True
        if target_type not in self.obj_hierarchy_info or len(self.obj_hierarchy_info[target_type]) == 0:
            return False
        else:
            for subtype in self.obj_hierarchy_info[target_type]:
                if self._is_valid_type(subtype, curr_type):
                    return True
            return False

    def _check_predicate_usage_pddl(self, pddl_snippet, predicate_list, action_params, part='preconditions'):
        """
        This function checks three types of errors:
            - check if the num of params given matches the num of params in predicate definition
            - check if there is any param that is not listed under `Parameters:`
            - check if the param type matches that in the predicate definition
        """
        def get_ordinal_suffix(_num):
            return {1: 'st', 2: 'nd', 3: 'rd'}.get(_num % 10, 'th') if _num not in (11, 12, 13) else 'th'

        pred_names = {predicate_list[i]['name']: i for i in range(len(predicate_list))}
        pddl_elems = [e for e in pddl_snippet.split(' ') if e != '']
        idx = 0
        while idx < len(pddl_elems):
            if pddl_elems[idx] == '(' and idx + 1 < len(pddl_elems):
                if pddl_elems[idx + 1] in pred_names:
                    curr_pred_name = pddl_elems[idx + 1]
                    curr_pred_params = list()
                    target_pred_info = predicate_list[pred_names[curr_pred_name]]
                    # read params
                    idx += 2
                    while idx < len(pddl_elems) and pddl_elems[idx] != ')':
                        curr_pred_params.append(pddl_elems[idx])
                        idx += 1
                    # check if the num of params are correct
                    n_expected_param = len(target_pred_info['params'])
                    if n_expected_param != len(curr_pred_params):
                        feedback_message = f'In the {part}, the predicate `{curr_pred_name}` requires {n_expected_param} parameters but {len(curr_pred_params)} parameters were provided. Please revise the PDDL model to fix this error.\n\nParameters:'
                        return False, 'invalid_predicate_usage', None, feedback_message
                    # check if there is any unknown param
                    for curr_param in curr_pred_params:
                        if curr_param not in action_params:
                            feedback_message = f'In the {part} and in the predicate `{curr_pred_name}`, there is an unknown parameter {curr_param}. You should define all parameters (i.e., name and type) under the `Parameters` list. Please revise the PDDL model to fix this error (and other potentially similar errors).\n\nParameters:'
                            return False, 'invalid_predicate_usage', None, feedback_message
                    # check if the object types are correct
                    target_param_types = [target_pred_info['params'][t_p] for t_p in target_pred_info['params']]
                    for param_idx, target_type in enumerate(target_param_types):
                        curr_param = curr_pred_params[param_idx]
                        claimed_type = action_params[curr_param]

                        if not self._is_valid_type(target_type, claimed_type):
                            feedback_message = f'There is a syntax error in the {part.lower()}, the {param_idx+1}-{get_ordinal_suffix(param_idx+1)} parameter of `{curr_pred_name}` should be a `{target_type}` but a `{claimed_type}` was given. Please use the correct predicate or devise new one(s) if needed (but note that you should use existing predicates as much as possible).\n\nParameters:'
                            return False, 'invalid_predicate_usage', None, feedback_message
            idx += 1
        return True, 'invalid_predicate_usage', None, None

    def check_predicate_usage(self, llm_output, **kwargs):
        """
        This function performs very basic check over whether the predicates are used in a valid way.
            This check should be performed at the end.
        """
        # parse predicates
        new_predicates = parse_new_predicates(llm_output)
        curr_predicates = deepcopy(kwargs['curr_predicates'])
        curr_predicates.extend(new_predicates)
        curr_predicates = parse_predicates(curr_predicates)

        # get action params
        params_info = parse_param_output(llm_output)

        # check preconditions
        precond_str = llm_output.split('\nPreconditions:')[1].split('```\n')[1].strip()
        precond_str = precond_str.replace('\n', ' ').replace('(', ' ( ').replace(')', ' ) ')
        validation_info = self._check_predicate_usage_pddl(precond_str, curr_predicates, params_info, part='preconditions')
        if not validation_info[0]:
            return validation_info

        eff_str = llm_output.split('\nEffects:')[1].split('```\n')[1].strip()
        eff_str = eff_str.replace('\n', ' ').replace('(', ' ( ').replace(')', ' ) ')
        return self._check_predicate_usage_pddl(eff_str, curr_predicates, params_info, part='effects')


def main():
    kwargs = {
        'curr_predicates': list()
    }
    obj_hierarchy = {
        "furnitureAppliance": [],
        "householdObject": ["smallReceptacle"]
    }

    pddl_snippet = """
Apologies for the confusion. Since the predicates are already defined, I will not list them under 'New Predicates'. Here is the revised PDDL model.

Parameters:
1. ?x - householdObject: the object to put in/on the furniture or appliance
2. ?y - furnitureAppliance: the furniture or appliance to put the object in/on

Preconditions:
```
(and
    (robot-at ?y)
    (robot-holding ?x)
    (pickupable ?x)
    (object-clear ?x)
    (or
        (not (openable ?y))
        (opened ?y)
    )
)
```

Effects:
```
(and
    (not (robot-holding ?x))
    (robot-hand-empty)
    (object-on ?x ?y)
    (if (openable ?y) (closed ?y))
)
```

New Predicates:
1. (closed ?y - furnitureAppliance): true if the furniture or appliance ?y is closed
2. (openable ?y - householdObject): true if the furniture or appliance ?y can be opened
3. (furnitureappliance ?x - furnitureAppliance): true if xxxxxxxxx
    """

    pddl_validator = PDDL_Syntax_Validator(obj_hierarchy)
    print(pddl_validator.check_unsupported_keywords(pddl_snippet, **kwargs))
    print(pddl_validator.check_messed_output(pddl_snippet, **kwargs))
    print(pddl_validator.check_param_types(pddl_snippet, **kwargs))
    print(pddl_validator.check_predicate_names(pddl_snippet, **kwargs))
    print(pddl_validator.check_predicate_format(pddl_snippet, **kwargs))
    print(pddl_validator.check_predicate_usage(pddl_snippet, **kwargs))
    # print(pddl_validator.perform_validation(pddl_snippet, **kwargs))


if __name__ == '__main__':
    main()
