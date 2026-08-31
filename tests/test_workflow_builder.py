import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow_builder import (
    DirectorValidationError,
    align_frames,
    build_prompt,
    duration_to_frames,
    effective_context_frames,
    normalize_project,
    prompt_with_continuity_contract,
)


def sample_project(mode="ref2v", accelerated=False, second_pass=False):
    lora = "ref2v_v0_1" if mode == "ref2v" else "fl2v_v1_1"
    return {
        "id": "test-project",
        "name": "Test",
        "settings": {
            "mode": mode,
            "aspect_ratio": "16_9",
            "megapixels": "0_6",
            "duration_seconds": 10,
            "acceleration_enabled": accelerated,
            "acceleration_lora": lora,
            "steps": 4 if accelerated else 20,
            "sampler": "euler" if accelerated else "res_multistep",
            "scheduler": "simple" if accelerated else "beta",
            "second_pass": second_pass,
            "output_quality": "720p",
            "sequence_mode": "continuous",
            "continuity": True,
        },
        "shots": [
            {
                "id": "s001",
                "title": "S001",
                "prompt": "<Picture 1> walks forward.",
                "assets": {"images": [{"path": "director/a.png", "name": "a.png"}], "videos": [], "audios": []},
            },
            {
                "id": "s002",
                "title": "S002",
                "prompt": "Continue the motion.",
                "assets": {"images": [{"path": "director/b.png", "name": "b.png"}], "videos": [], "audios": []},
            },
        ],
    }


class WorkflowBuilderTests(unittest.TestCase):
    def test_duration_uses_h3_frame_grid(self):
        self.assertEqual(align_frames(240), 243)
        self.assertEqual(duration_to_frames(10), 243)
        normalized = normalize_project(sample_project())
        self.assertEqual(normalized["settings"]["frames"], 243)
        self.assertEqual(normalized["settings"]["width"], 1056)
        self.assertEqual(normalized["settings"]["height"], 608)

    def test_continuity_normalizes_all_delivered_clip_lengths(self):
        project = sample_project("ref2v")
        first = build_prompt(project, "s001")
        second = build_prompt(project, "s002")

        def node_of(result, class_type):
            return next(node for node in result.prompt.values()
                        if node["class_type"] == class_type)

        first_encoder = node_of(first, "MiniMaxH3ReferenceToVideo")
        second_encoder = node_of(second, "MiniMaxH3ReferenceToVideo")
        first_trim = node_of(first, "H3DirectorTrimAV")
        second_context = node_of(second, "H3DirectorGeneratedAVMaskedContext")
        second_motion = node_of(second, "H3DirectorMotionCondition")

        self.assertEqual(first_encoder["inputs"]["length"], 243)
        self.assertEqual(first_trim["inputs"]["trim_frames"], 5)
        self.assertEqual(second_encoder["inputs"]["length"], 277)
        self.assertEqual(second_context["inputs"]["context_length"], 39)
        self.assertEqual(second_context["inputs"]["video_feather_tokens"], 0)
        self.assertEqual(second_motion["inputs"]["context_length"], 39)
        # S001: 243-5; S002+: 277-39. Both deliver exactly 238 frames.
        self.assertEqual(243 - 5, 277 - 39)

    def test_context_frame_budget_uses_the_plugins_actual_grid(self):
        self.assertEqual(effective_context_frames(22), 39)
        self.assertEqual(effective_context_frames(39), 39)
        self.assertEqual(effective_context_frames(90), 90)
        self.assertEqual(effective_context_frames(20, "motion_context"), 5)
        self.assertEqual(effective_context_frames(22, "motion_context"), 22)
        self.assertEqual(effective_context_frames(0), 0)

        project = sample_project("ref2v")
        project["settings"]["context_frames"] = 90
        second = build_prompt(project, "s002")
        encoder = next(node for node in second.prompt.values()
                       if node["class_type"] == "MiniMaxH3ReferenceToVideo")
        context = next(node for node in second.prompt.values()
                       if node["class_type"] == "H3DirectorGeneratedAVMaskedContext")
        self.assertEqual(encoder["inputs"]["length"], 277)
        self.assertEqual(context["inputs"]["context_length"], 39)
        self.assertEqual(243 - 5, 277 - 39)

    def test_continuous_mode_disables_video_feather(self):
        project = sample_project("ref2v")
        project["settings"]["video_feather_tokens"] = 99
        normalized = normalize_project(project)
        self.assertEqual(normalized["settings"]["video_feather_tokens"], 0)
        second = build_prompt(normalized, "s002")
        context = next(node for node in second.prompt.values()
                       if node["class_type"] == "H3DirectorGeneratedAVMaskedContext")
        self.assertEqual(context["inputs"]["video_feather_tokens"], 0)

    def test_independent_shots_do_not_load_continuity_nodes(self):
        project = sample_project("ref2v")
        project["settings"]["sequence_mode"] = "shots"
        normalized = normalize_project(project)
        self.assertFalse(normalized["settings"]["continuity"])
        result = build_prompt(normalized, "s002")
        types = {node["class_type"] for node in result.prompt.values()}
        self.assertNotIn("H3DirectorLoadAVLatent", types)
        self.assertNotIn("H3DirectorMotionCondition", types)
        self.assertNotIn("H3DirectorGeneratedAVMaskedContext", types)

    def test_aspect_megapixels_and_output_quality_are_resolved(self):
        project = sample_project(second_pass=True)
        project["settings"].update({"aspect_ratio": "9_16", "megapixels": "0_4", "output_quality": "1080p"})
        settings = normalize_project(project)["settings"]
        self.assertEqual((settings["width"], settings["height"]), (480, 864))
        self.assertEqual((settings["target_width"], settings["target_height"]), (1088, 1920))

    def test_text_to_video_uses_prompt_only_without_accelerators(self):
        project = sample_project("t2v")
        result = build_prompt(project, "s001")
        types = {node["class_type"] for node in result.prompt.values()}
        self.assertIn("JZL_MiniMaxH3ImageToVideoDual", types)
        self.assertNotIn("LoadImage", types)
        self.assertNotIn("LoraLoaderModelOnly", types)
        self.assertNotIn("TESpeedMiniMaxH3", types)
        self.assertTrue(result.warnings)

    def test_image_to_video_requires_first_frame_and_accepts_acceleration(self):
        project = sample_project("i2v", accelerated=True)
        result = build_prompt(project, "s001")
        types = {node["class_type"] for node in result.prompt.values()}
        self.assertIn("JZL_MiniMaxH3ImageToVideoDual", types)
        self.assertIn("LoraLoaderModelOnly", types)
        project["shots"][0]["assets"]["images"] = []
        with self.assertRaisesRegex(DirectorValidationError, "首帧"):
            build_prompt(project, "s001")

    def test_reference_mode_uses_reference_encoder(self):
        result = build_prompt(sample_project("ref2v"), "s001")
        types = {node["class_type"] for node in result.prompt.values()}
        self.assertIn("MiniMaxH3ReferenceToVideo", types)
        self.assertNotIn("LoraLoaderModelOnly", types)

    def test_fl_v11_is_available_as_experimental_reference_acceleration(self):
        project = sample_project("ref2v", accelerated=True)
        project["settings"]["acceleration_lora"] = "fl2v_v1_1"
        result = build_prompt(project, "s001")
        self.assertTrue(any("跨模式实验" in warning for warning in result.warnings))
        self.assertIn("LoraLoaderModelOnly", {node["class_type"] for node in result.prompt.values()})

    def test_reference_lora_is_rejected_for_image_mode(self):
        project = sample_project("i2v", accelerated=True)
        project["settings"]["acceleration_lora"] = "ref2v_v0_1"
        with self.assertRaisesRegex(DirectorValidationError, "不兼容"):
            build_prompt(project, "s001")

    def test_disabling_acceleration_restores_default_sampling(self):
        project = sample_project("i2v", accelerated=False)
        project["settings"].update({"steps": 4, "sampler": "euler", "scheduler": "simple"})
        settings = normalize_project(project)["settings"]
        self.assertEqual(settings["steps"], 20)
        self.assertEqual(settings["sampler"], "res_multistep")
        self.assertEqual(settings["scheduler"], "beta")

    def test_second_pass_and_continuation_are_connected(self):
        result = build_prompt(sample_project("ref2v", second_pass=True), "s002")
        types = [node["class_type"] for node in result.prompt.values()]
        self.assertIn("MinimaxH3LatentUpscaler3D", types)
        self.assertIn("JZL_MiniMaxH3CondSync", types)
        self.assertIn("H3DirectorMotionCondition", types)
        self.assertIn("H3DirectorGeneratedAVMaskedContext", types)
        self.assertEqual(types.count("SamplerCustomAdvanced"), 2)

    def test_second_pass_uses_exact_quality_dimensions_across_base_megapixels(self):
        geometries = []
        for megapixels in ("0_4", "0_6"):
            project = sample_project("ref2v", second_pass=True)
            project["settings"]["megapixels"] = megapixels
            normalized = normalize_project(project)
            result = build_prompt(normalized, "s002")
            upscaler = next(
                node for node in result.prompt.values()
                if node["class_type"] == "MinimaxH3LatentUpscaler3D"
            )
            inputs = upscaler["inputs"]
            self.assertEqual(inputs["mode"], "target dimensions")
            self.assertFalse(inputs["keep_proportion"])
            geometries.append((inputs["mode.width"], inputs["mode.height"]))
        self.assertEqual(geometries, [(1280, 736), (1280, 736)])

    def test_second_pass_can_inherit_existing_continuity_dimensions(self):
        result = build_prompt(
            sample_project("ref2v", second_pass=True),
            "s002",
            target_dimensions_override=(1312, 752),
        )
        upscaler = next(
            node for node in result.prompt.values()
            if node["class_type"] == "MinimaxH3LatentUpscaler3D"
        )
        self.assertEqual(upscaler["inputs"]["mode.width"], 1312)
        self.assertEqual(upscaler["inputs"]["mode.height"], 752)
        self.assertEqual(upscaler["inputs"]["align"], 16)
        self.assertTrue(any("1312×752" in warning for warning in result.warnings))

    def test_continuous_mode_forces_masked_latent_strategy(self):
        project = sample_project("ref2v")
        project["settings"].update({"continuity_strategy": "motion_context", "context_frames": 22})
        normalized = normalize_project(project)
        self.assertEqual(normalized["settings"]["continuity_strategy"], "masked_latent")
        self.assertEqual(normalized["settings"]["context_frames"], 39)
        result = build_prompt(normalized, "s002")
        types = {node["class_type"] for node in result.prompt.values()}
        self.assertIn("H3DirectorMotionCondition", types)
        self.assertIn("H3DirectorLoadAVLatent", types)
        self.assertIn("H3DirectorGeneratedAVMaskedContext", types)
        self.assertIn("H3DirectorLatentTailFrame", types)
        self.assertNotIn("MiniMaxH3MotionContext", types)

    def test_continuous_ref2v_prepends_inherited_composition_anchor(self):
        project = sample_project("ref2v")
        project["shots"][1]["prompt"] = (
            "subject_definitions:\n<Subject 1> matches <Picture 1>.\n\n"
            "detailed_description:\nKeep <Picture 2> behind her."
        )
        result = build_prompt(project, "s002")
        nodes = result.prompt
        anchor_id, _anchor = next(
            (node_id, node) for node_id, node in nodes.items()
            if node["class_type"] == "H3DirectorLatentTailFrame"
        )
        encoder = next(node for node in nodes.values()
                       if node["class_type"] == "MiniMaxH3ReferenceToVideo")
        self.assertEqual(encoder["inputs"]["ref_images.ref_image_0"], [anchor_id, 0])
        self.assertIn("ref_images.ref_image_1", encoder["inputs"])
        self.assertIn("<Picture 1> is the exact final delivered frame", encoder["inputs"]["prompt"])
        self.assertIn("matches <Picture 2>", encoder["inputs"]["prompt"])
        self.assertIn("Keep <Picture 3> behind her", encoder["inputs"]["prompt"])

    def test_continuation_prompt_contract_is_injected_into_h3_section(self):
        prompt = "subject_definitions:\n<Subject 1> is a woman.\n\ndetailed_description:\n[Shot 1] She walks."
        settings = {
            "sequence_mode": "continuous",
            "continuity": True,
            "continuity_prompt_lock": True,
        }
        result = prompt_with_continuity_contract(prompt, 1, settings)
        self.assertIn("detailed_description:\nThis segment is not a new shot", result)
        self.assertIn("Do not cut", result)
        self.assertIn("throughout every delivered frame", result)
        self.assertIn("Do not perform an action reserved for a later shot", result)
        self.assertIn("dissolve", result)
        self.assertIn("first twelve delivered frames", result)
        self.assertIn("Do not repeat or reset an action already completed", result)
        self.assertEqual(prompt_with_continuity_contract(prompt, 0, settings), prompt)

    def test_legacy_continuity_switch_migrates_to_sequence_mode(self):
        project = sample_project()
        project["settings"].pop("sequence_mode")
        project["settings"]["continuity"] = False
        settings = normalize_project(project)["settings"]
        self.assertEqual(settings["sequence_mode"], "shots")
        self.assertFalse(settings["continuity"])

    def test_legacy_mode_and_frames_are_migrated(self):
        project = sample_project()
        project["settings"] = {"mode": "fl2v_v1_1", "frames": 243, "width": 1024, "height": 576}
        settings = normalize_project(project)["settings"]
        self.assertEqual(settings["mode"], "i2v")
        self.assertTrue(settings["acceleration_enabled"])
        self.assertEqual(settings["acceleration_lora"], "fl2v_v1_1")
        self.assertAlmostEqual(settings["duration_seconds"], 10.125)
        self.assertEqual(settings["aspect_ratio"], "16_9")
        self.assertEqual(settings["megapixels"], "0_6")

    def test_unsafe_asset_path_is_rejected(self):
        project = sample_project()
        project["shots"][0]["assets"]["images"][0]["path"] = "../escape.png"
        with self.assertRaises(DirectorValidationError):
            normalize_project(project)


if __name__ == "__main__":
    unittest.main()
