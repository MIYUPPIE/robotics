"""YarnGPT-backed Nigerian-language voice prompts -- pre-rendered only.

kiosk_app/hand_ui/registration_app/admin_app may import voice.cache and
voice.playback (local, cached-audio only). voice.client (the network API
wrapper) must only ever be imported by voice/generate_prompts.py -- enforced
by voice/tests/test_no_live_tts_in_runtime.py.
"""
