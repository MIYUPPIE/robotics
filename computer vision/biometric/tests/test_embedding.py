from biometric.embedding import select_provider


class TestSelectProvider:
    def test_prefers_cuda_when_available(self):
        providers = select_provider(["CPUExecutionProvider", "CUDAExecutionProvider"])
        assert providers[0] == "CUDAExecutionProvider"
        assert "CPUExecutionProvider" in providers

    def test_falls_back_to_cpu_only(self):
        assert select_provider(["CPUExecutionProvider"]) == ["CPUExecutionProvider"]

    def test_explicit_preference_wins_when_available(self):
        providers = select_provider(
            ["CPUExecutionProvider", "CUDAExecutionProvider"], preferred="CPUExecutionProvider"
        )
        assert providers == ["CPUExecutionProvider"]

    def test_unavailable_preference_falls_back_to_default_priority(self):
        providers = select_provider(
            ["CPUExecutionProvider", "CUDAExecutionProvider"], preferred="TensorrtExecutionProvider"
        )
        assert providers[0] == "CUDAExecutionProvider"
