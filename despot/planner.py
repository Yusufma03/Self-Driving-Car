from despot import *

def load_data():
    with open('poses.json', 'r') as fin:
        parsed = json.load(fin)
    return parsed

def get_agent_poses(dic, index):
    ret = [
        v[index][:2]
        for k, v in dic.items()
    ]
    return ret

if __name__=='__main__':
    data = load_data()
    agent_poses = get_agent_poses(data, 0)
    robot_pos = [120, 0]
    out = build_despot(robot_pos, agent_poses)
    action = planning(out)
    print(action)