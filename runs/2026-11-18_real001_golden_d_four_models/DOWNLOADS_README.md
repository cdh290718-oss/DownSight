# real001, t=29.0s: SMPL-X parameter downloads

01_new_four_models: current full-2D golden D, four Sapiens2-1B detectors.
02_legacy_three_methods: historical original and v2 (one body per camera), plus historical D3d baseline (one shared body).
03_legacy_head_perturbations: historical D3d with positive/negative Head-anchor perturbations.

All NPZ files contain only fitted parameters and small metadata; no SMPL-X licensed model weights, images or raw meshes. JSON companions contain the same coefficients. Body pose is (1,63), each hand pose (1,45), global orientation (1,3), translation (1,3); rotations are axis-angle in radians and translation is meters. Gender neutral, use_pca=False; flat_hand_mean is stored per file and must be respected. Zero local hand input does not imply a flat hand when flat_hand_mean=False.

Load NPZ with allow_pickle=False. Pass betas, body_pose, global_orient, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression and transl to a matching SMPL-X layer. You must obtain the licensed SMPL-X model separately. Metadata keys are not model arguments.

shared_cam3_cv means one physical body is fitted to both images and saved in cam3 coordinates. To render in cam4, transform the regenerated mesh using camera_transform.json. Do NOT fit or add the two camera bodies together. Legacy original/v2 files are separate per-view fits and already use their named camera's coordinates.

Historical original/v2 overlays include MANO hand replacement/seam processing. SMPL-X coefficients reconstruct the SMPL-X body surface, not those extra hand mesh edits. Current golden D fixes local hand pose and does not use WiLoR observations. No MPJPE, PA-MPJPE or GT-based RMSE is claimed; this recording has no body GT. Camera and Head anchors retain the historical estimated/calibrated provenance and have not been independently validated against true body geometry.
