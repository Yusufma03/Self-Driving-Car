cd ../ros-lanechanging/autocar/scripts
python generate_poses.py
cp poses.json ../../../despot/poses.json
cd ../../../despot
python planner.py
mv cmds.json ../ros-lanechanging/autocar/scripts/
