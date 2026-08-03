import unittest

from ezlocalai.VIDEO_UTILS import (
    choose_video_gpu_residency,
    is_pythagorean_theorem_request,
    ltx_frame_count_for_duration,
    make_music_video_storyboard_image,
    make_pythagorean_equation_overlay_image,
    plan_music_video_scenes,
)


class LtxScenePlanningTests(unittest.TestCase):
    def test_frame_count_rounds_up_to_ltx_8n_plus_1(self):
        self.assertEqual(ltx_frame_count_for_duration(5, 24), 121)
        self.assertEqual(ltx_frame_count_for_duration(10, 24), 241)
        self.assertEqual(ltx_frame_count_for_duration(20, 24), 481)
        self.assertEqual(ltx_frame_count_for_duration(5.05, 24), 129)

    def test_music_video_scene_plan_splits_song_into_short_scenes(self):
        scenes = plan_music_video_scenes(
            duration=12,
            scene_duration=5,
            frame_rate=24,
            max_scene_duration=20,
        )

        self.assertEqual(len(scenes), 3)
        self.assertEqual([s["duration"] for s in scenes], [5, 5, 2])
        self.assertEqual([s["num_frames"] for s in scenes], [121, 121, 49])

    def test_music_video_scene_duration_is_capped(self):
        scenes = plan_music_video_scenes(
            duration=45,
            scene_duration=30,
            frame_rate=24,
            max_scene_duration=20,
        )

        self.assertEqual([s["duration"] for s in scenes], [20, 20, 5])
        self.assertEqual([s["num_frames"] for s in scenes], [481, 481, 121])

    def test_two_minute_music_video_plan_uses_six_ltx_scenes(self):
        scenes = plan_music_video_scenes(
            duration=120,
            scene_duration=20,
            frame_rate=24,
            max_scene_duration=20,
        )

        self.assertEqual(len(scenes), 6)
        self.assertEqual([s["start"] for s in scenes], [0, 20, 40, 60, 80, 100])
        self.assertTrue(all(s["num_frames"] == 481 for s in scenes))

    def test_storyboard_image_matches_requested_size_and_is_not_blank(self):
        image = make_music_video_storyboard_image(
            prompt="Heavy metal song about the Pythagorean theorem",
            lyrics="[Chorus]\nC squared lights the hypotenuse",
            scene_index=0,
            scene_count=4,
            size="512x320",
            keyscale="E minor",
        )

        self.assertEqual(image.size, (512, 320))
        self.assertGreater(len(image.getcolors(maxcolors=1_000_000)), 20)

    def test_pythagorean_request_detection_matches_prompt_or_lyrics(self):
        self.assertTrue(
            is_pythagorean_theorem_request(
                prompt="Heavy metal song about the Pythagorean theorem"
            )
        )
        self.assertTrue(
            is_pythagorean_theorem_request(
                lyrics="[Chorus]\nC squared lights the hypotenuse"
            )
        )
        self.assertFalse(
            is_pythagorean_theorem_request(
                prompt="Heavy metal song about prime numbers",
                lyrics="The factorization burns",
            )
        )

    def test_pythagorean_overlay_matches_requested_size_and_has_alpha(self):
        image = make_pythagorean_equation_overlay_image(size="512x320")

        self.assertEqual(image.mode, "RGBA")
        self.assertEqual(image.size, (512, 320))
        self.assertIsNotNone(image.getchannel("A").getbbox())
        self.assertGreater(len(image.getcolors(maxcolors=1_000_000)), 20)

    def test_video_gpu_residency_prefers_model_offload_when_vram_is_freed(self):
        self.assertEqual(
            choose_video_gpu_residency(
                configured_mode="auto",
                total_gb=23.5,
                free_gb=19.0,
                text_encoder_on_gpu=True,
            ),
            "model_offload",
        )

    def test_video_gpu_residency_keeps_sequential_on_3090_after_bnb_load(self):
        self.assertEqual(
            choose_video_gpu_residency(
                configured_mode="auto",
                total_gb=23.5,
                free_gb=11.2,
                text_encoder_on_gpu=True,
            ),
            "sequential",
        )

    def test_video_gpu_residency_keeps_short_clips_sequential_on_3090(self):
        self.assertEqual(
            choose_video_gpu_residency(
                configured_mode="auto",
                total_gb=23.5,
                free_gb=11.2,
                text_encoder_on_gpu=True,
                num_frames=65,
                width=512,
                height=320,
            ),
            "sequential",
        )

    def test_video_gpu_residency_uses_model_offload_for_short_clips_with_headroom(self):
        self.assertEqual(
            choose_video_gpu_residency(
                configured_mode="auto",
                total_gb=23.5,
                free_gb=12.5,
                text_encoder_on_gpu=True,
                num_frames=65,
                width=512,
                height=320,
            ),
            "model_offload",
        )

    def test_video_gpu_residency_keeps_long_scenes_sequential_on_3090(self):
        self.assertEqual(
            choose_video_gpu_residency(
                configured_mode="auto",
                total_gb=23.5,
                free_gb=11.2,
                text_encoder_on_gpu=True,
                num_frames=241,
                width=512,
                height=320,
            ),
            "sequential",
        )

    def test_video_gpu_residency_keeps_sequential_for_constrained_startup(self):
        self.assertEqual(
            choose_video_gpu_residency(
                configured_mode="auto",
                total_gb=23.5,
                free_gb=3.6,
                text_encoder_on_gpu=False,
            ),
            "sequential",
        )

    def test_video_gpu_residency_honors_full_override(self):
        self.assertEqual(
            choose_video_gpu_residency(
                configured_mode="full",
                total_gb=23.5,
                free_gb=19.0,
                text_encoder_on_gpu=True,
            ),
            "full",
        )


if __name__ == "__main__":
    unittest.main()
