"""MQTT alert routing to zone ESP8266 nodes.

Fire/weapon alerts are safety-critical and go to a `critical` topic per
zone; falls are a staff notification, not a siren; theft is logged for
human review rather than triggering anything automatic -- see each
detector module's docstring for why.
"""
from __future__ import annotations

import json
import logging
from enum import StrEnum
from typing import Any

import paho.mqtt.client as mqtt
from config import MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_CLIENT_ID

logger = logging.getLogger(__name__)


class AlertType(StrEnum):
    WEAPON = "weapon"
    FIRE = "fire"
    FALL = "fall"
    THEFT = "theft"


def _topic_for(alert_type: AlertType, zone: str | None) -> str:
    zone = zone or "all"
    if alert_type in (AlertType.WEAPON, AlertType.FIRE):
        return f"store/alerts/critical/{zone}"
    if alert_type is AlertType.FALL:
        return "store/alerts/staff"
    return f"store/alerts/review/{zone}"  # AlertType.THEFT


class AlertPublisher:
    """Thin wrapper around an MQTT client.

    Pass a fake client in tests -- anything with `.connect(host, port)` and
    `.publish(topic, payload)` methods works, no real broker needed.
    """

    def __init__(self, client: Any | None = None):
        self.client = client or mqtt.Client(client_id=MQTT_CLIENT_ID)
        self._connected = False

    def connect(self) -> None:
        try:
            self.client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT)
            self._connected = True
        except OSError:
            logger.exception(
                "MQTT broker unreachable at %s:%s -- alerts will be logged only, not delivered",
                MQTT_BROKER_HOST,
                MQTT_BROKER_PORT,
            )
            self._connected = False

    def send_alert(self, alert_type: AlertType, zone: str | None = None, **payload: Any) -> None:
        topic = _topic_for(alert_type, zone)
        message = json.dumps({"type": alert_type.value, "zone": zone, **payload})
        if not self._connected:
            logger.warning("MQTT not connected, dropping alert: %s %s", topic, message)
            return
        self.client.publish(topic, message)
        logger.info("published alert: %s %s", topic, message)
