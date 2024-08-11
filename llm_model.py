import os
import traceback

from config import get_openai_key

import openai
from openai import OpenAI
from wrapt_timeout_decorator import timeout
import tenacity


@timeout(dec_timeout=30, use_signals=False)
def connect_openai(client, engine, messages, temperature, max_tokens,
                   top_p, frequency_penalty, presence_penalty, stop):
    try:
        return client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop
        )
    except Exception as e:
        print(f'[ERROR] OpenAI model error: {e}')
        raise e


class GPT_Chat:
    def __init__(self, engine, stop=None, max_tokens=1000, temperature=0, top_p=1,
                 frequency_penalty=0.0, presence_penalty=0.0):
        self.engine = engine
        print(get_openai_key())
        self.gpt_client = OpenAI(api_key=get_openai_key())
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.freq_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop

    def get_response(self, prompt, messages=None, max_retry=5, verbose=False):
        conn_success, llm_output = False, ''
        if messages is not None:
            messages = messages
        else:
            messages = [{'role': 'user', 'content': prompt}]

        try:
            r = tenacity.Retrying(
                stop=tenacity.stop_after_attempt(max_retry),
                wait=tenacity.wait_fixed(1.5),
                reraise=True
            )
            response = r.__call__(
                connect_openai,
                client=self.gpt_client,
                engine=self.engine,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.freq_penalty,
                presence_penalty=self.presence_penalty,
                stop=self.stop)
            llm_output = response.choices[0].message.content
            if verbose:
                print(f'[INFO] Connection success - token usage: {response.usage}')
            conn_success = True
        except Exception:
            print(traceback.format_exc())

        return conn_success, llm_output


def main():
    llm_gpt = GPT_Chat(engine='gpt-4')
    _, llm_output = llm_gpt.get_response('test')
    print(llm_output)


if __name__ == '__main__':
    main()


