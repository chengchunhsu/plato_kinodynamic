import torch
import numpy as np
import os
import plotly.graph_objects as go
import mediapy as media
from PIL import Image
from plato_copilot.kinodynamics.dynamics_model import DynamicsDeltaOutputModel, safe_device

transformations = torch.load('poke_side_transformations.pt')
save_video = True

positions_list = []
positions_list.append(np.array([[0, 0, 0]]))
poses_list = []
poses_list.append(np.array([0, 0, 0, 0, 0, 0]))
actions_list = [np.array([[-0.1, 0., -0.001]]) * 0.05]
for t in transformations:
    # t[0][3] = -t[0][3]
    pos = np.hstack((positions_list[-1], np.ones((positions_list[-1].shape[0], 1))))
    pos = t @ pos.T
    translation = pos[:3].T
    pose = np.zeros(6)
    pose[:3] = translation
    positions_list.append(translation)
    poses_list.append(pose)
    action = np.array([[-0.1, 0., -0.001]]) * 0.05
    actions_list.append(action)

poses = torch.tensor(poses_list)
actions = torch.tensor(actions_list)

experiment_path = "../experiments/curved_train/run_001"
dynamics_model = safe_device(DynamicsDeltaOutputModel())
dynamics_model.load_state_dict(torch.load(os.path.join(experiment_path, "dynamics_model_delta.pth")))
dynamics_model.eval()

max_len = 10
predicted_y = [np.zeros((1, 6))] * 10
dynamics_model.use_low_noise()
for i in range(len(poses_list) - max_len):
    object_pose = poses[i:i+max_len].unsqueeze(0)
    action = actions[i:i+max_len].unsqueeze(0)
    data = {
        "x_seq": object_pose.float(),
        "a_seq": action.float()
    }
    y = dynamics_model.predict(data)
    # predicted_y.append(np.zeros((1, 6)))
    predicted_y.append(y[0].detach().cpu().numpy())

# from plato_copilot.vision.plotly_utils import plotly_draw_3d_pcd
# plotly_draw_3d_pcd(poses)
# plotly_draw_3d_pcd(np.squeeze(np.stack(predicted_y))[:, :3])

# Compute axis ranges
all_positions = np.vstack(positions_list)
all_predicted = np.array(predicted_y).squeeze(1)[:, :3]
all_positions = np.vstack((all_positions, all_predicted))
# x_min, x_max = all_positions[:,0].min(), all_positions[:,0].max()
# y_min, y_max = all_positions[:,1].min(), all_positions[:,1].max()
# z_min, z_max = all_positions[:,2].min(), all_positions[:,2].max()
x_min, x_max = -0.1, 0.02
y_min, y_max = -0.02, 0.1
z_min, z_max = -0.02, 0.1

# Create initial data
data = [go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=10, color="blue"), name="Estimated position"),
        go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=3, color="blue"), name="Estimated trajectory"),
        go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=10, color="red"), name="Predicted position"),
        go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=3, color="red"), name="Predicted trajectory")]

# Create frames
frames = []
for i, (positions_i, predictions_i) in enumerate(zip(positions_list, predicted_y)):
    estimated_traj = positions_list[:i+1]
    predicted_traj = predicted_y[:i]
    frame = go.Frame(
        data=[go.Scatter3d(
                  x=positions_i[:, 0],
                  y=positions_i[:, 1],
                  z=positions_i[:, 2],
                  mode='markers',
                  marker=dict(size=10, color="blue")),
              go.Scatter3d(
                  x=[t[0][0] for t in estimated_traj],
                  y=[t[0][1] for t in estimated_traj],
                  z=[t[0][2] for t in estimated_traj],
                  mode='markers',
                  marker=dict(size=3, color="blue")),
              go.Scatter3d(
                  x=predictions_i[:, 0],
                  y=predictions_i[:, 1],
                  z=predictions_i[:, 2],
                  mode='markers',
                  marker=dict(size=10, color="red")),
              go.Scatter3d(
                  x=[t[0][0] for t in predicted_traj],
                  y=[t[0][1] for t in predicted_traj],
                  z=[t[0][2] for t in predicted_traj],
                  mode='markers',
                  marker=dict(size=3, color="red")),
        ],
        name=str(i+1),
    )
    frames.append(frame)

scene = dict(
            xaxis=dict(range=[x_min, x_max], autorange=False, title_font=dict(size=16)),
            yaxis=dict(range=[y_min, y_max], autorange=False, title_font=dict(size=16)),
            zaxis=dict(range=[z_min, z_max], autorange=False, title_font=dict(size=16)),
            aspectmode="cube",
            camera={
                    "eye": {"x": 0.3, "y": -1.5, "z": 1.5},  # Camera location
                    "center": {"x": 0, "y": -0.2, "z": 0},       # Focus on the origin
                    "up": {"x": 0, "y": 0, "z": 1},           # Z is up
                },
        )

fig = go.Figure(data=data, frames=frames)
# Set axis ranges
fig.update_layout(
    scene=scene,
    legend=dict(
            font=dict(
            size=18,
        )
    ),
    updatemenus=[dict(
        type='buttons',
        showactive=False,
        buttons=[dict(
            label='Play',
            method='animate',
            args=[None, dict(frame=dict(duration=500, redraw=True),
                             fromcurrent=True,
                             transition=dict(duration=0))]
        )]
    )]
)

# Add slider
sliders = [dict(
    steps=[dict(
        method='animate',
        args=[[str(k+1)],
              dict(mode='immediate',
                   frame=dict(duration=0, redraw=True),
                   transition=dict(duration=0))],
        label=str(k+1)
    ) for k in range(len(frames))],
    transition=dict(duration=0),
    x=0,  # slider starting position
    y=0,
    currentvalue=dict(font=dict(size=12), prefix='Frame: ', visible=True, xanchor='center'),
    len=1.0
)]

fig.update_layout(sliders=sliders)
fig.show()

if save_video:
    # Create figure
    fig = go.Figure(data=data)
    fig.update_layout(
        scene=scene,
        legend=dict(
                font=dict(
                size=18,
            )
        )
    )

    # Save each frame as an image
    image_frames = []
    for i, frame in enumerate(frames):
        fig.update(data=frame.data)  # Update figure with frame data
        filename = os.path.join(experiment_path, f"frame_{i}.png")
        fig.write_image(filename, width=1000, height=800)  # Save as image
        img = Image.open(filename).convert("RGB")
        image_frames.append(np.array(img))  # Read the image back into memory

    # Create a video from the image frames using mediapy
    video_filename = os.path.join(experiment_path, f"kinodynamic_vis_plato.mp4")
    media.write_video(video_filename, image_frames, fps=20)  # Save video at 10 frames per second
