"""Small launcher node for the H3 Director Console sidebar."""


class H3DirectorConsole:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_id": ("STRING", {"default": "director-project", "multiline": False}),
                "open_director_console": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "打开导演台",
                        "label_off": "打开导演台",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("project_id",)
    FUNCTION = "execute"
    CATEGORY = "MiniMax H3/Director Console"

    def execute(self, project_id, open_director_console=False):
        return (project_id,)


from .continuity_nodes import (
    NODE_CLASS_MAPPINGS as CONTINUITY_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as CONTINUITY_NODE_DISPLAY_NAME_MAPPINGS,
)


NODE_CLASS_MAPPINGS = {
    "H3DirectorConsole": H3DirectorConsole,
    **CONTINUITY_NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3DirectorConsole": "🎬 MiniMax H3 导演台",
    **CONTINUITY_NODE_DISPLAY_NAME_MAPPINGS,
}
