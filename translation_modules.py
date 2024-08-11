import os
import re
from copy import deepcopy


class PDDL_Translator:
    def __init__(self, domain, engine, pddl_to_language_prompt=None):
        self.domain = domain
        self.engine = engine
        self.pddl_to_language_prompt = pddl_to_language_prompt
        # use the default prompt
        if pddl_to_language_prompt is None:
            with open(os.path.join('prompts/common', f'pddl_to_language_prompt.txt')) as f:
                self.pddl_to_language_prompt = f.read().strip()
        # init LLM
        if 'text-' in engine:
            raise NotImplementedError
        else:
            from llm_model import GPT_Chat
            self.llm_conn = GPT_Chat(engine=engine)

    def pddl_to_language(self, pddl, predicate_list, contextual_info=None):
        relevant_predicates = list()
        # extract relevant predicates (to make the prompt shorter)
        for pred in predicate_list:
            pred_name = pred['name']
            if pred_name in pddl:
                relevant_predicates.append(pred)
        prompt = str(self.pddl_to_language_prompt).strip() + '\n\n'
        prompt += '\nPredicates:'
        for i, pred in enumerate(relevant_predicates):
            prompt += f'\n{i+1}. '
            prompt += relevant_predicates[i]['raw']
        prompt += f'\n\nContextual information:'
        if contextual_info is None:
            prompt += ' N/A'
        else:
            prompt += f' {contextual_info}'
        prompt += f'\n\nPDDL:\n```\n{pddl.strip()}\n```\n'
        prompt += '\nTranslated PDDL:'
        # get the output from llm
        conn_success, llm_output = self.llm_conn.get_response(prompt)
        return conn_success, llm_output

    def translate_domain_model(self, domain_model, predicate_list, action_info=None):
        domain_model = deepcopy(domain_model)
        for act in domain_model:
            contextual_info = None if action_info is None else action_info[act]['desc']
            # translate precondition
            print(f'[INFO] translating preconditions of `{act}`')
            precond_pddl = domain_model[act]['preconditions']
            _, llm_output = self.pddl_to_language(precond_pddl, predicate_list,
                                                  contextual_info=contextual_info)
            precond_list = list()
            # drop the leading '(and' and the ending ')'
            for precond in llm_output.split('\n')[1:-1]:
                precond_item = precond[:-1].lower().strip()
                precond_list.append(precond_item)
            domain_model[act]['translated_preconditions'] = precond_list

            # translate effects
            print(f'[INFO] translating effects of `{act}`')
            eff_pddl = domain_model[act]['effects']
            _, llm_output = self.pddl_to_language(eff_pddl, predicate_list,
                                                  contextual_info=contextual_info)
            eff_list = list()
            # drop the leading '(and' and the ending ')'
            for eff in llm_output.split('\n')[1:-1]:
                eff_item = eff[:-1].lower().strip()
                eff_list.append(eff_item)
            domain_model[act]['translated_effects'] = eff_list
        return domain_model


class Action_Description_Generator:
    def __init__(self, domain, engine, action_desc_prompt=None):
        self.domain = domain
        self.engine = engine
        self.action_desc_prompt = action_desc_prompt
        if action_desc_prompt is None:
            with open(os.path.join('prompts/common', f'action_description_prompt.txt')) as f:
                self.action_desc_prompt = f.read().strip()
            with open(os.path.join(f'prompts/{domain}', f'domain_desc.txt')) as f:
                domain_desc_str = f.read().strip()
                if '{domain_desc}' in self.action_desc_prompt:
                    self.action_desc_prompt = self.action_desc_prompt.replace('{domain_desc}', domain_desc_str)
        # init LLM
        if 'text-' in engine:
            raise NotImplementedError
        else:
            from llm_model import GPT_Chat
            self.llm_conn = GPT_Chat(engine=engine)

    @staticmethod
    def _extract_objects(str_example):
        str_example = str_example.lower().strip()
        objs = [s for s in str_example.split(' ') if '_' in s]
        return objs

    def _validate_desc(self, short_description, short_example, param_names):
        # check if all the parameters are included in the short description
        missing_params = list()
        for p_name in param_names:
            if p_name not in short_description:
                missing_params.append(p_name)
        if len(missing_params) > 0:
            feedback_msg = 'The following parameter(s) are not mentioned in the short description:'
            for p_i, p_name in enumerate(missing_params):
                feedback_msg += f'\n{p_i + 1}. {p_name}'
            feedback_msg += f'\n\nPlease note that you need to include all the parameters in the short description and short example. Please revise them.\n\nShort description:'
            return False, 'missing_params_in_description', feedback_msg
        # check if the param values in the short example is valid
        example_objs = self._extract_objects(short_example)
        for obj in example_objs:
            obj_prefix, obj_suffix = obj.split('_')[0], obj.split('_')[1]
            if obj_prefix.isdigit() or not obj_suffix.isdigit():
                feedback_msg = f'In the short example, `{obj}` is not a valid parameter value. The parameter values should be in the format of' + '`{letters}_{numbers}`.'
                feedback_msg += f' Note that if `{obj}` is not supposed to be a parameter value, please don\'t use `_` in it.'
                feedback_msg += f'\n\nPlease fix this error (and other potentially similar errors) by coming up with valid parameter values.\n\nShort description:'
                return False, 'missing_params_in_description', feedback_msg
        return True, None, None

    def get_act_description(self, act_params, detailed_description, max_iter=5):
        act_desc_prompt = str(self.action_desc_prompt) + '\n\n'
        act_desc_prompt += 'Action: ' + detailed_description
        act_desc_prompt += '\n\n' + 'Parameters:'
        for param_i, param in enumerate(act_params):
            act_desc_prompt += f'\n{param_i+1}. {param}'
        act_desc_prompt += '\n\nShort description:'

        param_names = [p.split(' ')[0] for p in act_params]
        messages = [{'role': 'user', 'content': act_desc_prompt}]
        i_iter, conn_success, llm_output = 0, False, ''
        while i_iter < max_iter:
            i_iter += 1
            # get the output from llm
            conn_success, llm_output = self.llm_conn.get_response(prompt=None, messages=messages)
            messages.append({'role': 'assistant', 'content': llm_output})
            # validate the description and short example
            try:
                short_description = llm_output.split('\n')[0].strip()
                short_example = llm_output.split('\n')[1].split(':')[1].strip()
            except:
                raise Exception(f'[ERROR] error in parsing the llm output: {llm_output}')
            validation_info = self._validate_desc(short_description, short_example, param_names)
            if not validation_info[0]:
                feedback_msg = validation_info[2]
                messages.append({'role': 'user', 'content': feedback_msg})
                print('-' * 10)
                print(feedback_msg)
                print('-' * 10)
                continue
            else:
                break
        short_description = llm_output.split('\n')[0].strip()
        short_example = llm_output.split('\n')[1].split(':')[1].strip()
        return conn_success, short_description, short_example



