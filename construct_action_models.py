import json
import os
from copy import deepcopy
from datetime import datetime

from addict import Dict

from translation_modules import PDDL_Translator
from utils.pddl_output_utils import parse_new_predicates, parse_full_domain_model
from pddl_syntax_validator import PDDL_Syntax_Validator


def get_action_prompt(prompt_template, action_desc, include_extra_info):
    action_desc_prompt = action_desc['desc']
    if include_extra_info:
        for feedback_i in action_desc['extra_info']:
            action_desc_prompt += ' ' + feedback_i
    full_prompt = str(prompt_template) + ' ' + action_desc_prompt
    return full_prompt, action_desc_prompt


def get_predicate_prompt(predicate_list):
    predicate_prompt = 'You can create and define new predicates, but you may also reuse the following predicates:'
    if len(predicate_list) == 0:
        predicate_prompt += '\nNo predicate has been defined yet'
    else:
        for i, p in enumerate(predicate_list):
            predicate_prompt += f'\n{i+1}. {p["raw"]}'
    return predicate_prompt


def construct_action_model(llm_conn, action_predicate_prompt,
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

    results_dict = Dict({action_name: Dict()})

    conn_success, llm_output = False, ''
    no_syntax_error = False
    messages = [{'role': 'user', 'content': action_predicate_prompt}]
    i_iter = 0
    while not no_syntax_error and i_iter < max_iteration:
        i_iter += 1
        print(f'[INFO] generating PDDL of action: `{action_name}` | # of messages: {len(messages)}')
        llm_message = _shorten_message(messages, i_iter) if shorten_message else messages
        conn_success, llm_output = llm_conn.get_response(prompt=None, messages=llm_message)
        messages.append({'role': 'assistant', 'content': llm_output})
        if not conn_success:
            raise Exception('Fail to connect to the LLM')
        results_dict[action_name][f'iter_{i_iter}']['llm_output'] = llm_output
        results_dict[action_name]['llm_output'] = llm_output
        print(llm_output)

        if syntax_validator is not None:
            val_kwargs = {'curr_predicates': predicate_list}
            validation_info = syntax_validator.perform_validation(llm_output, **val_kwargs)
            if not validation_info[0]:
                error_type, feedback_msg = validation_info[1], validation_info[3]
                print('-' * 20)
                print(f'[INFO] feedback message on {error_type}:')
                print(feedback_msg)
                results_dict[action_name][f'iter_{i_iter}']['error_type'] = error_type
                results_dict[action_name][f'iter_{i_iter}']['feedback_msg'] = feedback_msg
                messages.append({'role': 'user', 'content': feedback_msg})
                print('-' * 20)
                continue
        no_syntax_error = True

    if not no_syntax_error:
        print(f'[WARNING] syntax error remaining in the action model: {action_name}')
    # update the predicate list
    new_predicates = parse_new_predicates(llm_output)
    predicate_list.extend(new_predicates)
    results_dict[action_name]['new_predicates'] = [new_p['raw'] for new_p in new_predicates]
    return llm_output, results_dict, predicate_list


def main():
    actions = None  # None means all actions
    skip_actions = None
    prompt_version = 'model_blocksworld'
    include_additional_info = True
    domain = 'logistics'  # 'household', 'logistics', 'tyreworld'
    engine = 'gpt-4'  # 'gpt-4' or 'gpt-3.5-turbo'
    unsupported_keywords = ['forall', 'when', 'exists', 'implies']
    max_iterations = 3 if ('gpt-4' in engine and domain != 'household') else 2  # we only do 2 iteration in Household because there are too many actions, so the experiments are expensive to run
    max_feedback = 8 if 'gpt-4' in engine else 3    # more feedback doesn't help with other models like gpt-3.5-turbo
    shorten_messages = False if 'gpt-4' in engine else True
    add_date_str = True

    pddl_prompt_dir = f'prompts/common/'
    domain_data_dir = f'prompts/{domain}'
    with open(os.path.join(pddl_prompt_dir, f'{prompt_version}/pddl_prompt.txt')) as f:
        prompt_template = f.read().strip()
    with open(os.path.join(domain_data_dir, f'domain_desc.txt')) as f:
        domain_desc_str = f.read().strip()
        if '{domain_desc}' in prompt_template:
            prompt_template = prompt_template.replace('{domain_desc}', domain_desc_str)
    with open(os.path.join(domain_data_dir, f'action_model.json')) as f:
        action_desc = json.load(f)
    with open(os.path.join(domain_data_dir, f'hierarchy_requirements.json')) as f:
        obj_hierarchy_info = json.load(f)['hierarchy']

    # only GPT-4 is able to revise PDDL models with feedback message
    syntax_validator = PDDL_Syntax_Validator(obj_hierarchy_info, unsupported_keywords=unsupported_keywords) if 'gpt-4' in engine else None
    pddl_translator = PDDL_Translator(domain, engine='gpt-4')

    if actions is None:
        actions = list(action_desc.keys())
    predicate_list = list()

    # init LLM
    from llm_model import GPT_Chat
    llm_gpt = GPT_Chat(engine=engine)

    results_dict = Dict()
    date_str = datetime.today().strftime('%Y-%m-%d-%H-%M') if add_date_str else ''
    result_log_dir = f'results/{date_str}/{domain}/{prompt_version}'
    os.makedirs(result_log_dir, exist_ok=True)

    for i_iter in range(max_iterations):
        readable_results = ''
        prev_predicate_list = deepcopy(predicate_list)

        for i_action, action in enumerate(actions):
            if skip_actions is not None and action in skip_actions:
                continue

            action_prompt, action_desc_prompt = get_action_prompt(prompt_template, action_desc[action],
                                                                  include_additional_info)
            print('\n')
            print('#' * 20)
            print(f'[INFO] iter {i_iter} | action {i_action}: {action}.')
            print('#' * 20)
            readable_results += '\n' * 2 + '#' * 20 + '\n' + f'Action: {action}\n' + '#' * 20 + '\n'

            predicate_prompt = get_predicate_prompt(predicate_list)
            results_dict[action]['predicate_prompt'] = predicate_prompt
            results_dict[action]['action_desc'] = action_desc_prompt
            readable_results += '-' * 20
            readable_results += f'\n{predicate_prompt}\n' + '-' * 20

            action_predicate_prompt = f'{action_prompt}\n\n{predicate_prompt}'
            action_predicate_prompt += '\n\nParameters:'
            pddl_construction_output = construct_action_model(llm_gpt, action_predicate_prompt, action, predicate_list,
                                                              shorten_message=shorten_messages, max_iteration=max_feedback,
                                                              syntax_validator=syntax_validator)
            llm_output, action_results_dict, predicate_list = pddl_construction_output

            results_dict.update(action_results_dict)
            readable_results += '\n' + '-' * 10 + '-' * 10 + '\n'
            readable_results += llm_output + '\n'

        readable_results += '\n' + '-' * 10 + '-' * 10 + '\n'
        readable_results += 'Extracted predicates:\n'
        for i, p in enumerate(predicate_list):
            readable_results += f'\n{i + 1}. {p["raw"]}'

        with open(os.path.join(result_log_dir, f'{engine}_0_{i_iter}.txt'), 'w') as f:
            f.write(readable_results)
        with open(os.path.join(result_log_dir, f'{engine}_0_{i_iter}.json'), 'w') as f:
            json.dump(results_dict, f, indent=4, sort_keys=False)

        gen_done = False
        if len(prev_predicate_list) == len(predicate_list):
            print(f'[INFO] iter {i_iter} | no new predicate has been defined, will terminate the process')
            gen_done = True

        if gen_done:
            break

    # finalize the pddl model and translate it
    parsed_domain_model = parse_full_domain_model(results_dict, actions)
    translated_domain_model = pddl_translator.translate_domain_model(parsed_domain_model, predicate_list, action_desc)
    # save the result
    with open(os.path.join(result_log_dir, f'{engine}_pddl.json'), 'w') as f:
        json.dump(translated_domain_model, f, indent=4, sort_keys=False)
    # save the predicates
    predicate_list_str = ''
    for idx, predicate in enumerate(predicate_list):
        if idx == 0:
            predicate_list_str += predicate['raw']
        else:
            predicate_list_str += '\n' + predicate['raw']
    with open(os.path.join(result_log_dir, f'{engine}_predicates.txt'), 'w') as f:
        f.write(predicate_list_str)
    # save the expr log for annotation
    simplified_result_dict = Dict()
    for act in results_dict:
        if act not in action_desc:
            continue
        simplified_result_dict[act]['action_desc'] = results_dict[act]['action_desc']
        simplified_result_dict[act]['llm_output'] = results_dict[act]['llm_output']
        simplified_result_dict[act]['translated_preconditions'] = translated_domain_model[act]['translated_preconditions']
        simplified_result_dict[act]['translated_effects'] = translated_domain_model[act]['translated_effects']
        simplified_result_dict[act]['annotation'] = list()
    with open(os.path.join(result_log_dir, f'{engine}_pddl_for_annotations.json'), 'w') as f:
        json.dump(simplified_result_dict, f, indent=4, sort_keys=False)


if __name__ == '__main__':
    main()
