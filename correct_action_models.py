import json
import os

from addict import Dict

from construct_action_models import get_predicate_prompt
from pddl_syntax_validator import PDDL_Syntax_Validator
from translation_modules import PDDL_Translator, Action_Description_Generator
from utils.pddl_output_utils import parse_new_predicates, parse_full_domain_model
from utils.predicate_reader import read_predicates


def correct_action_model(llm_conn, messages,
                         action_name, predicate_list, max_iteration=8,
                         shorten_message=False, syntax_validator=None):
    def _shorten_message(_msg, _step_i):
        """
        Only keep the latest LLM output and correction feedback
        """
        print(f'[INFO] step: {_step_i} | num of messages: {len(_msg)}')
        if _step_i == 1:
            return [_msg[0]]
        else:
            _short_msg = [_msg[0], _msg[2 * (_step_i - 1) - 1], _msg[2 * (_step_i - 1)]]
            assert _short_msg[1]['role'] == 'assistant'
            assert _short_msg[2]['role'] == 'user'
            return _short_msg

    print(f'[INFO] correcting PDDL of action: `{action_name}`')
    print('-' * 20)
    print(messages[1]['content'])
    print('-' * 20)
    print(messages[2]['content'])
    print('-' * 20)

    user_mark_success = False
    conn_success, llm_output = False, ''
    no_syntax_error = False
    i_iter = 0
    while not no_syntax_error and i_iter < max_iteration:
        i_iter += 1
        llm_message = _shorten_message(messages, i_iter) if shorten_message else messages
        conn_success, llm_output = llm_conn.get_response(prompt=None,
                                                         messages=llm_message)
        messages.append({'role': 'assistant', 'content': llm_output})
        if not conn_success:
            raise Exception('Fail to connect to the LLM')
        print('-' * 20)
        print(llm_output)
        print('-' * 20)

        if syntax_validator is not None:
            val_kwargs = {'curr_predicates': predicate_list}
            validation_info = syntax_validator.perform_validation(llm_output, **val_kwargs)
            if not validation_info[0]:
                error_type, feedback_msg = validation_info[1], validation_info[3]
                print('-' * 20)
                print(f'[INFO] feedback message on {error_type}:')
                print(feedback_msg)
                print('-' * 20)
                messages.append({'role': 'user', 'content': feedback_msg})
                continue

        # get additional user feedback here
        additional_feedback = input('Additional feedback (use ; to separate multiple feedback messages, and click Enter if no additional feedback is needed):\n')
        if additional_feedback.strip() == '':
            user_mark_success = True
            break
        else:
            additional_feedback = additional_feedback.split(';')
            feedback_message = get_feedback_message(additional_feedback)
            messages.append({'role': 'user', 'content': feedback_message})
            print('-' * 20)
            print(feedback_message)
            print('-' * 20)
            continue

    # update the predicate list
    new_predicates = parse_new_predicates(llm_output)
    predicate_list.extend(new_predicates)
    return messages, predicate_list, llm_output, user_mark_success


def get_feedback_message(feedback_list):
    if len(feedback_list) == 1:
        feedback_str = 'There is an error in the PDDL model:'
    else:
        feedback_str = 'There are some errors in the PDDL model:'
    feedback_idx = 0
    for _, feedback in enumerate(feedback_list):
        if feedback.strip() == '':
            continue
        feedback_str += f'\n{feedback_idx+1}. {feedback}'
        feedback_idx += 1
    feedback_str += '\n\nPlease revise the PDDL model (and the list of predicates if needed) to fix the above errors (and other potentially similar errors).'
    feedback_str += '\n\nParameters:'
    return feedback_str


def main():
    actions = None  # None means all actions
    skip_actions = None
    prompt_version = 'model_blocksworld'
    domain = 'tyreworld'
    engine = 'gpt-4'  # 'gpt-4', 'text-davinci-003', 'gpt-3.5-turbo'
    unsupported_keywords = ['forall', 'when', 'exists', 'implies']
    max_feedback = 8 if 'gpt-4' in engine else 3  # more feedback doesn't help with other models like gpt-3.5-turbo
    shorten_messages = False if 'gpt-4' in engine else True
    result_log_dir = f'results/{domain}/{prompt_version}/corrected'
    os.makedirs(result_log_dir, exist_ok=True)

    expr_result_dir = f'results/{domain}/{prompt_version}'
    predicate_file = f'{engine}_predicates.txt'
    annotated_pddl_file = f'{engine}_pddl_for_annotations.json'
    with open(os.path.join(expr_result_dir, annotated_pddl_file)) as f:
        annotated_pddl = Dict(json.load(f))
    if actions is None:
        actions = list(annotated_pddl.keys())
    predicate_list = read_predicates(os.path.join(expr_result_dir, predicate_file), contain_idx=False)

    pddl_prompt_dir = f'prompts/common/'
    domain_data_dir = f'prompts/{domain}'
    with open(os.path.join(pddl_prompt_dir, f'{prompt_version}/pddl_prompt.txt')) as f:
        prompt_template = f.read().strip()
    with open(os.path.join(domain_data_dir, f'domain_desc.txt')) as f:
        domain_desc_str = f.read().strip()
        if '{domain_desc}' in prompt_template:
            prompt_template = prompt_template.replace('{domain_desc}', domain_desc_str)
    with open(os.path.join(domain_data_dir, f'action_model.json')) as f:
        action_info = json.load(f)
    with open(os.path.join(domain_data_dir, f'hierarchy_requirements.json')) as f:
        obj_hierarchy_info = json.load(f)['hierarchy']

    # only GPT-4 is able to revise PDDL models with feedback message
    syntax_validator = PDDL_Syntax_Validator(obj_hierarchy_info, unsupported_keywords=unsupported_keywords) if 'gpt-4' in engine else None
    pddl_translator = PDDL_Translator(domain, engine='gpt-4')
    act_desc_generator = Action_Description_Generator(domain, engine='gpt-3.5-turbo')

    # init LLM
    if 'text-' in engine:
        raise NotImplementedError
    else:
        from llm_model import GPT_Chat
        llm_gpt = GPT_Chat(engine=engine)

    results_dict = Dict()
    for i_action, action in enumerate(actions):
        if skip_actions is not None and action in skip_actions:
            print(f'[INFO] skipping: {action}')
            continue
        if len(annotated_pddl[action]['annotation']) == 0:
            print(f'[INFO] skipping: {action}')
            continue
        if action in results_dict and results_dict[action]['user_mark_success']:
            print(f'[INFO] skipping {action}')
            continue

        print('#' * 20)
        print('#' * 20)
        print(f'[INFO] action {i_action}: `{action}`.')
        print('#' * 20)
        print('#' * 20)

        action_prompt = str(prompt_template) + ' ' + annotated_pddl[action]['action_desc']
        predicate_prompt = get_predicate_prompt(predicate_list)
        action_predicate_prompt = f'{action_prompt}\n\n{predicate_prompt}'
        action_predicate_prompt += '\n\nParameters:'
        init_messages = [{'role': 'user', 'content': action_predicate_prompt},
                         {'role': 'assistant', 'content': annotated_pddl[action]['llm_output']},
                         {'role': 'user', 'content': get_feedback_message(annotated_pddl[action]['annotation'])}]

        correction_info = correct_action_model(llm_gpt, init_messages, action, predicate_list,
                                               max_iteration=max_feedback,
                                               shorten_message=shorten_messages, syntax_validator=syntax_validator)
        messages, predicate_list, llm_output, user_mark_success = correction_info
        results_dict[action]['messages'] = messages
        results_dict[action]['llm_output'] = llm_output
        results_dict[action]['user_mark_success'] = user_mark_success

    with open(os.path.join(result_log_dir, f'{engine}_correction.json'), 'w') as f:
        json.dump(results_dict, f, indent=4, sort_keys=False)

    # save into readable format
    result_readable = ''
    for action in results_dict:
        result_readable += '#' * 20 + '\n' + f'Action: {action}\n' + '#' * 20 + '\n'
        for message in results_dict[action]['messages']:
            result_readable += '-' * 20 + '\n'
            result_readable += message['role'] + ':\n'
            result_readable += message['content'] + '\n'
            result_readable += '-' * 20 + '\n'
    with open(os.path.join(result_log_dir, f'{engine}_correction.txt'), 'w') as f:
        f.write(result_readable)

    # merge domain model info
    for act in annotated_pddl:
        if act not in results_dict:
            results_dict[act]['llm_output'] = str(annotated_pddl[act]['llm_output'])
            results_dict[act]['action_desc'] = str(annotated_pddl[act]['action_desc'])

    # save the predicates
    predicate_list_str = ''
    for idx, predicate in enumerate(predicate_list):
        if idx == 0:
            predicate_list_str += predicate['raw']
        else:
            predicate_list_str += '\n' + predicate['raw']
    with open(os.path.join(result_log_dir, f'{engine}_final_predicates.txt'), 'w') as f:
        f.write(predicate_list_str)

    # finalize the pddl model and translate it
    parsed_domain_model = parse_full_domain_model(results_dict, actions)
    translated_domain_model = pddl_translator.translate_domain_model(parsed_domain_model, predicate_list, action_info)
    """
    for act in translated_domain_model:
        act_params = translated_domain_model[act]['parameters']
        act_detailed_desc = translated_domain_model[act]['action_desc']
        print(f'[INFO] generating short description and short example for the action: `{act}`')
        _, short_description, short_example = act_desc_generator.get_act_description(act_params, act_detailed_desc)
        print(f'[INFO] short description: {short_description}')
        print(f'[INFO] short example: {short_example}')
        translated_domain_model[act]['short'] = short_description
        translated_domain_model[act]['prompt_example'] = short_example
    """
    with open(os.path.join(result_log_dir, f'{engine}_final_pddl.json'), 'w') as f:
        json.dump(translated_domain_model, f, indent=4, sort_keys=False)


if __name__ == '__main__':
    main()
