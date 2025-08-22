import numpy as np
import plotly.graph_objects as go

from plato_copilot.vision.point_cloud_utils import *

class JengaTowerState():
    def __init__(self, layers=6, blocks_per_layer=3, block_length=3.0, block_width=1.0, block_height=0.5):
        self.layers = layers
        self.blocks_per_layer = blocks_per_layer
        self.block_length = block_length
        self.block_width = block_width
        self.block_height = block_height

        self.layout = np.empty((self.layers * self.blocks_per_layer, 3))
        self.offset = np.zeros((self.layers * self.blocks_per_layer, 3))
        self.init_layout()

    def init_layout(self):
        for k in range(1, self.layers + 1):
            if k % 2 == 1:
                for i in range(self.blocks_per_layer):
                    id = i + (k - 1) * self.blocks_per_layer
                    self.layout[id] = [i + 1, 0, k]
            else:
                for j in range(self.blocks_per_layer):
                    id = j + (k - 1) * self.blocks_per_layer
                    self.layout[id] = [0, j + 1, k]

    def reset_layout(self, layout_matrix):
        # layout_matrix: MAX_Height x 3 binary matrix
        # 1 0 1
        # 0 1 0
        # 1 0 1
        layout_matrix = np.where(layout_matrix > 0, 1, 0)
        total_blocks = np.sum(layout_matrix)
        self.layout = np.empty((total_blocks, 3))
        # find the index of 1 from the matrix
        indices = np.where(layout_matrix == 1)
        for i in range(total_blocks):
            if (indices[0][i] + 1) % 2 == 1:
                self.layout[i] = [indices[1][i] + 1, 0, indices[0][i] + 1]
            else:
                self.layout[i] = [0, indices[1][i] + 1, indices[0][i] + 1]

    def plotly_jenga_3d_mesh(self, x, y, z, block_length, block_width, block_height, centroid_x_offset=0,
                             centroid_y_offset=0, centroid_yaw=0, orientation='horizontal', color='tan'):
        # Create a single Jenga block at the ah ha ha zspecified coordinates with given orientation
        # if orientation == 'horizontal':
        #     x0, x1 = x - block_length / 2, x + block_length / 2
        #     y0, y1 = y - block_width / 2, y + block_width / 2
        # else:  # vertical orientation
        #     x0, x1 = x - block_width / 2, x + block_width / 2
        #     y0, y1 = y - block_length / 2, y + block_length / 2

        yaw_angle = centroid_yaw
        if orientation == 'horizontal':
            yaw_angle += np.pi / 2
        rotaiton_matrix = np.array([[np.cos(yaw_angle), -np.sin(yaw_angle)], [np.sin(yaw_angle), np.cos(yaw_angle)]])
        delta_x, delta_y = np.dot(rotaiton_matrix, np.array([block_width / 2, block_length / 2]))
        x0, x1 = x + centroid_x_offset - delta_x, x + centroid_x_offset + delta_x
        y0, y1 = y + centroid_y_offset - delta_y, y + centroid_y_offset + delta_y

        z0, z1 = z - block_height / 2, z + block_height / 2
        return go.Mesh3d(
            # Define the vertices of the block
            x=[x0, x1, x1, x0, x0, x1, x1, x0],
            y=[y0, y0, y1, y1, y0, y0, y1, y1],
            z=[z0, z0, z0, z0, z1, z1, z1, z1],
            color=color,
            opacity=0.6,
            alphahull=0
        )

    def get_axes(self, axis_length=5):
        axes = [
            go.Scatter3d(
                x=[0, axis_length], y=[0, 0], z=[0, 0], mode='lines+text', name='X-axis',
                line=dict(color='red', width=4),
                text=["", "X"], textposition="top center"
            ),
            go.Scatter3d(
                x=[0, 0], y=[0, axis_length], z=[0, 0], mode='lines+text', name='Y-axis',
                line=dict(color='green', width=4),
                text=["", "Y"], textposition="top center"
            ),
            go.Scatter3d(
                x=[0, 0], y=[0, 0], z=[0, axis_length], mode='lines+text', name='Z-axis',
                line=dict(color='blue', width=4),
                text=["", "Z"], textposition="top center"
            )
        ]
        return axes

    def get_block_coordinates(self, block_id):
        return self.layout[block_id]

    def get_block_faces(self, block_id):
        i, j, k = self.layout[block_id]
        centroid_x_offset, centroid_y_offset, centroid_yaw = self.offset[block_id]
        if k % 2 == 1:
            x = i * self.block_width - self.block_width / 2
            y = self.block_length / 2
            z = (k - 1) * self.block_height + self.block_height / 2
        else:
            x = self.block_length / 2
            y = j * self.block_width - self.block_width / 2
            z = (k - 1) * self.block_height + self.block_height / 2

        orientation = 'horizontal' if k % 2 == 0 else 'vertical'

        yaw_angle = centroid_yaw
        if orientation == 'horizontal':
            yaw_angle += np.pi / 2
        rotaiton_matrix = np.array([[np.cos(yaw_angle), -np.sin(yaw_angle)], [np.sin(yaw_angle), np.cos(yaw_angle)]])
        delta_x, delta_y = np.dot(rotaiton_matrix, np.array([self.block_width / 2, self.block_length / 2]))
        x0, x1 = x + centroid_x_offset - delta_x, x + centroid_x_offset + delta_x
        y0, y1 = y + centroid_y_offset - delta_y, y + centroid_y_offset + delta_y

        z0, z1 = z - self.block_height / 2, z + self.block_height / 2

        x = [x0, x1, x1, x0, x0, x1, x1, x0]
        y = [y0, y0, y1, y1, y0, y0, y1, y1]
        z = [z0, z0, z0, z0, z1, z1, z1, z1]

        verts = []
        for i in range(8):
            verts.append((x[i], y[i], z[i]))

        return verts, generate_faces(verts, num_points_u=100, num_points_v=100)


    def visualize(self):
        fig = go.Figure()
        print(len(self.offset), len(self.layout))
        for offset, layout in zip(self.offset, self.layout):
            i, j, k = layout
            off_x, off_y, off_yaw = offset
            if k % 2 == 1:
                x = i * self.block_width - self.block_width / 2
                y = self.block_length / 2
                z = (k - 1) * self.block_height + self.block_height / 2
            else:
                x = self.block_length / 2
                y = j * self.block_width - self.block_width / 2
                z = (k - 1) * self.block_height + self.block_height / 2

            fig.add_trace(self.plotly_jenga_3d_mesh(x, y, z,
                                                    self.block_length, self.block_width, self.block_height,
                                                    centroid_x_offset=off_x, centroid_y_offset=off_y,
                                                    centroid_yaw=off_yaw,
                                                    orientation='horizontal' if k % 2 == 0 else 'vertical',
                                                    color="tan" if (k) % 2 == 1 else "gray"))

        axes = self.get_axes(axis_length=5)
        fig.add_traces(axes)

        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False),
                yaxis=dict(showbackground=False, showticklabels=False),
                zaxis=dict(showbackground=False, showticklabels=False),
                aspectmode='data'
            ),
            width=600,
            height=400,
            title='3D Visualization of a Jenga Tower'
        )
        fig.show()

    def hierarchical_visualize(self, selected_id):
        same_layer_color = "tan"
        upper_level = "cyan"
        lower_level = "gray"
        selected_color = "magenta"

        selected_i, selected_j, selected_k = self.layout[selected_id]
        fig = go.Figure()
        for layout in self.layout:
            i, j, k = layout
            if k % 2 == 1:
                x = i * self.block_width - self.block_width / 2
                y = self.block_length / 2
                z = (k - 1) * self.block_height + self.block_height / 2
            else:
                x = self.block_length / 2
                y = j * self.block_width - self.block_width / 2
                z = (k - 1) * self.block_height + self.block_height / 2

            if i == selected_i and j == selected_j and k == selected_k:
                color = selected_color
            elif k == selected_k:
                color = same_layer_color
            elif k > selected_k:
                color = upper_level
            else:
                color = lower_level

            fig.add_trace(self.plotly_jenga_3d_mesh(x, y, z, self.block_length, self.block_width, self.block_height,
                                                    orientation='horizontal' if k % 2 == 0 else 'vertical', color=color,
                                                    ))

        axes = self.get_axes(axis_length=5)
        fig.add_traces(axes)

        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False),
                yaxis=dict(showbackground=False, showticklabels=False),
                zaxis=dict(showbackground=False, showticklabels=False),
                aspectmode='cube'
            ),
            width=600,
            height=400,
            title='3D Visualization of a Jenga Tower'
        )
        fig.show()

    def highlight_visualize(self, selected_id):
        normal_color = "white"
        highlight_color = "magenta"

        selected_i, selected_j, selected_k = self.layout[selected_id]
        fig = go.Figure()
        for layout in self.layout:
            i, j, k = layout
            if k % 2 == 1:
                x = i * self.block_width - self.block_width / 2
                y = self.block_length / 2
                z = (k - 1) * self.block_height + self.block_height / 2
            else:
                x = self.block_length / 2
                y = j * self.block_width - self.block_width / 2
                z = (k - 1) * self.block_height + self.block_height / 2

            if i == selected_i and j == selected_j and k == selected_k:
                color = highlight_color
            else:
                color = normal_color

            fig.add_trace(self.plotly_jenga_3d_mesh(x, y, z, self.block_length, self.block_width, self.block_height,
                                                    orientation='horizontal' if k % 2 == 0 else 'vertical', color=color,
                                                    ))

        axes = self.get_axes(axis_length=5)
        fig.add_traces(axes)
        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False),
                yaxis=dict(showbackground=False, showticklabels=False),
                zaxis=dict(showbackground=False, showticklabels=False),
                aspectmode='cube'
            ),
            width=600,
            height=400,
            title='3D Visualization of a Jenga Tower'
        )
        fig.show()

    def move_block(self, id, new_i, new_j, new_k):
        if new_k % 2 == 1 and new_i == 0:
            raise ValueError('Invalid move: the layout is invalid')
        if new_k % 2 == 0 and new_j == 0:
            raise ValueError('Invalid move: the layout is invalid')
        for i in range(self.layers * self.blocks_per_layer):
            if self.layout[i][0] == new_i and self.layout[i][1] == new_j and self.layout[i][2] == new_k:
                raise ValueError('Invalid move: the target position is already occupied')
        self.layout[id] = [new_i, new_j, new_k]

    def update_single_block_state(self, id, delta_x, delta_y, delta_yaw):
        self.offset[id] = [delta_x, delta_y, delta_yaw]
