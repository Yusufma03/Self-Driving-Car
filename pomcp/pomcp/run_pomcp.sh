cd ../ros-lanechanging/autocar/scripts
python generate_poses.py
cp poses.json ../../../pomcp/poses.json
cd ../../../pomcp
pwd
python pomcp_planner.py