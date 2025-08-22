



class Observer():
    def __init__(self, type, config):
        self.type = type
        assert(self.type in ["video", "image", "camera"])

        self.config = config

        # make sure config for a type is defined
        assert(self.type in self.config)
        
        self.latest_rgbd = None

    def get_observation(self):
        if self.type == "video":
            return self.get_video_observation()
        elif self.type == "image":
            return self.get_image_observation()
        elif self.type == "camera":
            return self.get_camera_observation()
        

    def terminate(self):
        return self.terminate