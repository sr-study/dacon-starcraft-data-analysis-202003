import numpy as np

from .extract_gamestates import GameState


class CameraState(GameState):
    def init(self):
        self.p0_camera_x = []
        self.p0_camera_y = []
        self.p1_camera_x = []
        self.p1_camera_y = []

    def update(self, game_id, time, player, species, event, event_contents):
        if event == 'Camera':
            camera_x, camera_y = CameraState.parse_at(event_contents)

            if player == 0:
                self.p0_camera_x.append(camera_x)
                self.p0_camera_y.append(camera_y)
            else:
                self.p1_camera_x.append(camera_x)
                self.p1_camera_y.append(camera_y)

    def to_dict(self):
        return {
            'p0_camera_x_var': np.var(self.p0_camera_x),
            'p0_camera_y_var': np.var(self.p0_camera_y),
            'p1_camera_x_var': np.var(self.p1_camera_x),
            'p1_camera_y_var': np.var(self.p1_camera_y),
        }

    @staticmethod
    def parse_at(event_contents):
        mid = event_contents.find(',', 4)
        x = float(event_contents[4:mid])
        y = float(event_contents[mid+2:-1])
        return x, y
