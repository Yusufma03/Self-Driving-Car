import random
import numpy as np


def ucb_action(mcts, current_node, greedy):
    best_actions = []
    best_q_value = -np.inf
    mapping = current_node.action_map

    N = mapping.total_visit_count
    log_n = np.log(N + 1)

    actions = list(mapping.entries.values())
    random.shuffle(actions)
    for action_entry in actions:

        if not action_entry.is_legal:
            continue

        current_q = action_entry.mean_q_value

        if not greedy:
            current_q += mcts.find_fast_ucb(N, action_entry.visit_count, log_n)

        if current_q >= best_q_value:
            if current_q > best_q_value:
                best_actions = []
            best_q_value = current_q
            best_actions.append(action_entry.get_action())

    return best_actions[len(best_actions)-1]