import os
import json
import random


def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(curr_dir, 'household_planning_tasks.json')) as f:
        planning_tasks = json.load(f)

    instructions = list(planning_tasks.keys())
    print('Instructions: ', instructions)

    random_task = random.choice(instructions)
    print(f'Random task: "{random_task}"')

    task_info = planning_tasks[random_task]

    # one instruction may have one or more test cases, in which object states may vary
    task_cases = list(task_info.keys())
    print('Task cases: ', task_cases)
    random_case = random.choice(task_cases)

    # one test case may have one or more scenes, in which states of task-irrelevant objects may vary
    task_scenes = list(task_info[random_case].keys())
    print('Task scenes: ', task_scenes)
    random_scene = random.choice(task_scenes)

    object_info = task_info[random_case][random_scene]['objects']
    object_list = list(object_info.keys())
    print('Object list: ', object_list)

    object_states = list()
    for obj in object_list:
        object_states.extend(object_info[obj]['state'])
    print('Object states: ', object_states)


if __name__ == '__main__':
    main()
