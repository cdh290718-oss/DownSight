"""Stable names for the current algorithm and the reproducible archived route."""
CURRENT = 'mano_guided_original_seam_20260929_v1'
LEGACY = 'session_hand6_full_pipeline_20260926_v1'
V2 = 'stereo_guided_original_seam_v2_20261104'
SUPPORTED = (CURRENT, LEGACY, V2)


def selected(config):
    profile = config.get('pipeline_profile', CURRENT)
    if profile not in SUPPORTED:
        raise ValueError('Unsupported pipeline_profile: ' + str(profile))
    if (profile == V2) != bool(config.get('stereo_v2', False)):
        raise ValueError('v2 profile and stereo_v2 flag must agree')
    return profile
