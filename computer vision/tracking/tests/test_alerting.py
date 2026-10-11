import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alerting import AlertPublisher, AlertType


class FakeMqttClient:
    def __init__(self, raise_on_connect: bool = False):
        self._raise_on_connect = raise_on_connect
        self.published: list[tuple[str, str]] = []

    def connect(self, host, port):
        if self._raise_on_connect:
            raise OSError("connection refused")

    def publish(self, topic, payload):
        self.published.append((topic, payload))


def _connected_publisher() -> tuple[AlertPublisher, FakeMqttClient]:
    client = FakeMqttClient()
    publisher = AlertPublisher(client=client)
    publisher.connect()
    return publisher, client


def test_weapon_alert_routes_to_critical_zone_topic():
    publisher, client = _connected_publisher()

    publisher.send_alert(AlertType.WEAPON, zone="zone_1", class_name="gun", confidence=0.9)

    topic, _ = client.published[0]
    assert topic == "store/alerts/critical/zone_1"


def test_fire_alert_routes_to_critical_zone_topic():
    publisher, client = _connected_publisher()

    publisher.send_alert(AlertType.FIRE, zone="zone_2", class_name="smoke", confidence=0.7)

    topic, _ = client.published[0]
    assert topic == "store/alerts/critical/zone_2"


def test_fall_alert_routes_to_staff_topic_regardless_of_zone():
    publisher, client = _connected_publisher()

    publisher.send_alert(AlertType.FALL, track_id=3)

    topic, _ = client.published[0]
    assert topic == "store/alerts/staff"


def test_theft_alert_routes_to_review_zone_topic():
    publisher, client = _connected_publisher()

    publisher.send_alert(
        AlertType.THEFT, zone="zone_1", track_id=7, class_name="soda_can", quantity=1
    )

    topic, _ = client.published[0]
    assert topic == "store/alerts/review/zone_1"


def test_missing_zone_defaults_to_all():
    publisher, client = _connected_publisher()

    publisher.send_alert(AlertType.WEAPON, class_name="gun", confidence=0.9)

    topic, _ = client.published[0]
    assert topic == "store/alerts/critical/all"


def test_payload_carries_type_zone_and_extra_fields_as_json():
    publisher, client = _connected_publisher()

    publisher.send_alert(AlertType.THEFT, zone="zone_1", track_id=7, quantity=2)

    _, payload = client.published[0]
    body = json.loads(payload)
    assert body == {"type": "theft", "zone": "zone_1", "track_id": 7, "quantity": 2}


def test_successful_publish_is_logged_at_info(caplog):
    import logging

    publisher, _ = _connected_publisher()

    with caplog.at_level(logging.INFO):
        publisher.send_alert(AlertType.FALL, track_id=3)

    assert "published alert" in caplog.text


def test_alert_is_not_published_when_broker_is_unreachable():
    client = FakeMqttClient(raise_on_connect=True)
    publisher = AlertPublisher(client=client)

    publisher.connect()
    publisher.send_alert(AlertType.FIRE, zone="zone_1")

    assert client.published == []


def test_alert_is_not_published_before_connect_is_called():
    client = FakeMqttClient()
    publisher = AlertPublisher(client=client)

    publisher.send_alert(AlertType.FIRE, zone="zone_1")

    assert client.published == []
