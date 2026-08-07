"""Lazy CK wrapper around VideoHelperSuite's maintained Video Combine node."""


def get_vhs_video_combine_class():
    # Avoid importing VideoHelperSuite during custom-node discovery. Its module
    # registers routes through PromptServer.instance, which is only guaranteed
    # after the ComfyUI server has initialized. By the time INPUT_TYPES or the
    # node function is requested, all custom-node mappings are available.
    import nodes

    video_combine = nodes.NODE_CLASS_MAPPINGS.get("VHS_VideoCombine")
    if video_combine is None:
        raise RuntimeError(
            "CK Video Combine requires the enabled comfyui-VideoHelperSuite node pack "
            "and its VHS_VideoCombine node."
        )
    return video_combine


class VideoCombineCK:
    CATEGORY = "CK Nodes/Video/Output"
    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    FUNCTION = "combine_video"
    DESCRIPTION = "VideoHelperSuite Video Combine with CK category and localization."

    @classmethod
    def INPUT_TYPES(cls):
        return get_vhs_video_combine_class().INPUT_TYPES()

    def combine_video(self, **kwargs):
        return get_vhs_video_combine_class()().combine_video(**kwargs)


NODE_CLASS_MAPPINGS = {
    "VHS_VideoCombineIsolated": VideoCombineCK,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_VideoCombineIsolated": "CK Video Combine",
}
