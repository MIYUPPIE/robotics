from admin_app.auth import PinGate, hash_pin, verify_pin


class TestHashAndVerify:
    def test_correct_pin_verifies(self):
        stored = hash_pin("1234")
        assert verify_pin("1234", stored)

    def test_wrong_pin_fails(self):
        stored = hash_pin("1234")
        assert not verify_pin("9999", stored)

    def test_different_salts_produce_different_hashes(self):
        assert hash_pin("1234") != hash_pin("1234")

    def test_malformed_stored_hash_fails_closed(self):
        assert not verify_pin("1234", "not-a-valid-hash")
        assert not verify_pin("1234", "")


class TestPinGate:
    def test_correct_pin_unlocks(self):
        gate = PinGate(hash_pin("1234"))
        assert gate.try_pin("1234")
        assert not gate.locked

    def test_locks_after_max_attempts(self):
        gate = PinGate(hash_pin("1234"), max_attempts=3)
        assert not gate.try_pin("0000")
        assert not gate.try_pin("1111")
        assert not gate.try_pin("2222")
        assert gate.locked

    def test_locked_gate_rejects_even_the_correct_pin(self):
        gate = PinGate(hash_pin("1234"), max_attempts=1)
        gate.try_pin("0000")
        assert gate.locked
        assert not gate.try_pin("1234")

    def test_attempt_counter_resets_are_not_implicit(self):
        gate = PinGate(hash_pin("1234"), max_attempts=3)
        gate.try_pin("0000")
        assert gate.attempts == 1
        gate.try_pin("1234")  # correct -- but attempts counter is not reset by design
        assert gate.attempts == 1
