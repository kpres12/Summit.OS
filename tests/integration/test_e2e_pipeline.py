"""
End-to-End Integration Tests for Summit.OS

Tests the full pipeline: detection → classification → fusion → intent → tasking.
Also tests security layer, mesh sync, and gRPC services.

Run: python -m pytest tests/integration/test_e2e_pipeline.py -v
"""
import sys
import os
import math
import time
import asyncio
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# ── AI/ML Pipeline Tests ───────────────────────────────────

class TestAIDetection:
    """Test the object detection pipeline."""

    def test_mock_detector_produces_detections(self):
        from packages.ai.detection import MockDetector
        det = MockDetector(num_detections=5, seed=42)
        result = det.detect("fake_image")
        assert len(result.detections) > 0
        assert result.model_name == "mock"
        assert result.inference_ms > 0

    def test_mock_detector_deterministic(self):
        from packages.ai.detection import MockDetector
        d1 = MockDetector(seed=42)
        d2 = MockDetector(seed=42)
        r1 = d1.detect("img")
        r2 = d2.detect("img")
        assert len(r1.detections) == len(r2.detections)
        for a, b in zip(r1.detections, r2.detections):
            assert a.class_name == b.class_name
            assert a.confidence == b.confidence

    def test_nms_removes_overlapping(self):
        from packages.ai.detection import Detection, BoundingBox, non_max_suppression
        dets = [
            Detection(0, "car", 0.9, BoundingBox(100, 100, 200, 200)),
            Detection(0, "car", 0.7, BoundingBox(105, 105, 205, 205)),  # Overlaps
            Detection(1, "person", 0.8, BoundingBox(300, 300, 400, 400)),
        ]
        kept = non_max_suppression(dets, iou_threshold=0.3)
        assert len(kept) == 2  # car(0.9) + person

    def test_bounding_box_iou(self):
        from packages.ai.detection import BoundingBox
        a = BoundingBox(0, 0, 100, 100)
        b = BoundingBox(50, 50, 150, 150)
        iou = a.iou(b)
        assert 0.1 < iou < 0.3  # Partial overlap

    def test_create_detector_auto(self):
        from packages.ai.detection import create_detector, MockDetector
        det = create_detector("auto")
        assert isinstance(det, MockDetector)  # No YOLO installed


class TestAIClassification:
    """Test entity classification."""

    def test_bayesian_updates_posterior(self):
        from packages.ai.classification import BayesianClassifier, Evidence
        bc = BayesianClassifier(classes=["A", "B", "C"])
        ev = Evidence(source="visual", feature_name="class_name", value="person")
        result = bc.update("entity_1", ev)
        assert result.entity_id == "entity_1"
        assert result.top_confidence > 0

    def test_rule_based_adsb(self):
        from packages.ai.classification import RuleBasedClassifier, Evidence
        rc = RuleBasedClassifier()
        ev = Evidence(source="adsb", feature_name="has_adsb", value=True)
        result = rc.classify("aircraft_1", [ev])
        assert result.top_class == "CIVILIAN_AIRCRAFT"
        assert result.top_confidence >= 0.9

    def test_taxonomy_has_classes(self):
        from packages.ai.classification import EntityTaxonomy
        classes = EntityTaxonomy.get_all_classes()
        assert len(classes) > 30
        assert "F-16" in classes
        assert "MQ-9" in classes

    def test_rule_based_squawk_emergency(self):
        from packages.ai.classification import RuleBasedClassifier, Evidence
        rc = RuleBasedClassifier()
        ev = Evidence(source="transponder", feature_name="squawk", value="7700")
        result = rc.classify("aircraft_2", [ev])
        assert result.top_class == "EMERGENCY_AIRCRAFT"


class TestAIAnomaly:
    """Test anomaly detection."""

    def test_zscore_normal_data(self):
        from packages.ai.anomaly import ZScoreDetector, TimeSeriesPoint
        det = ZScoreDetector(window_size=20, z_threshold=3.0)
        # Feed normal data
        for i in range(20):
            result = det.ingest("e1", "speed", TimeSeriesPoint(value=50.0 + i * 0.1))
        # Normal point
        result = det.ingest("e1", "speed", TimeSeriesPoint(value=51.0))
        assert not result.is_anomaly

    def test_zscore_detects_anomaly(self):
        from packages.ai.anomaly import ZScoreDetector, TimeSeriesPoint
        det = ZScoreDetector(window_size=20, z_threshold=2.0)
        for i in range(20):
            det.ingest("e1", "speed", TimeSeriesPoint(value=50.0))
        # Extreme outlier
        result = det.ingest("e1", "speed", TimeSeriesPoint(value=200.0))
        assert result.is_anomaly
        assert result.score > 0.5

    def test_ema_detector(self):
        from packages.ai.anomaly import MovingAverageDetector, TimeSeriesPoint
        det = MovingAverageDetector()
        for i in range(20):
            det.ingest("e1", "alt", TimeSeriesPoint(value=1000.0))
        result = det.ingest("e1", "alt", TimeSeriesPoint(value=1001.0))
        assert result.detector_name == "ema"

    def test_ensemble_voting(self):
        from packages.ai.anomaly import EnsembleDetector, ZScoreDetector, MovingAverageDetector, TimeSeriesPoint
        det = EnsembleDetector(
            detectors=[ZScoreDetector(), MovingAverageDetector()],
            vote_threshold=2,
        )
        for i in range(20):
            det.ingest("e1", "speed", TimeSeriesPoint(value=50.0))
        result = det.ingest("e1", "speed", TimeSeriesPoint(value=51.0))
        assert result.detector_name == "ensemble"


class TestAIIntent:
    """Test intent prediction."""

    def test_trajectory_prediction(self):
        from packages.ai.intent import TrajectoryPredictor, Kinematics, Position
        tp = TrajectoryPredictor()
        for i in range(5):
            tp.update("e1", Kinematics(
                position=Position(lat=34.0 + i * 0.001, lon=-118.0, timestamp=i),
                velocity=(111.0, 0.0, 0.0),
                speed=111.0,
                heading=0.0,
            ))
        predicted = tp.predict("e1", horizon_s=60, steps=3)
        assert len(predicted) == 3
        assert predicted[0].lat > 34.004  # Moving north

    def test_threat_assessment_friendly(self):
        from packages.ai.intent import ThreatAssessor, Position, ThreatLevel
        ta = ThreatAssessor()
        level, score = ta.assess("e1", {}, Position(34.0, -118.0), is_friendly=True)
        assert level == ThreatLevel.NONE
        assert score == 0.0


# ── Security Tests ──────────────────────────────────────────

class TestSecurity:
    """Test security layer."""

    def test_jwt_issue_verify(self):
        from packages.security.auth import JWTAuth
        auth = JWTAuth(secret="test-secret")
        token = auth.issue("user1", roles=["OPERATOR"], scopes=["read", "write"])
        result = auth.verify(token)
        assert result.authenticated
        assert result.identity == "user1"
        assert "OPERATOR" in result.roles

    def test_jwt_revocation(self):
        from packages.security.auth import JWTAuth
        auth = JWTAuth(secret="test-secret")
        token = auth.issue("user1")
        assert auth.verify(token).authenticated
        auth.revoke(token)
        assert not auth.verify(token).authenticated

    def test_api_key_lifecycle(self):
        from packages.security.auth import APIKeyAuth
        auth = APIKeyAuth()
        raw_key, api_key = auth.create_key("user1", scopes=["read"])
        result = auth.verify(raw_key)
        assert result.authenticated
        assert result.identity == "user1"
        auth.revoke(api_key.key_id)
        assert not auth.verify(raw_key).authenticated

    def test_rbac_hierarchy(self):
        from packages.security.rbac import RBACEngine, Action, Resource
        rbac = RBACEngine()
        rbac.assign_role("user1", "OPERATOR")
        # Operator inherits VIEWER permissions
        assert rbac.check_permission("user1", Action.READ, Resource.ENTITIES)
        assert rbac.check_permission("user1", Action.WRITE, Resource.TASKS)
        # But cannot admin users
        assert not rbac.check_permission("user1", Action.ADMIN, Resource.USERS)

    def test_rbac_classification_levels(self):
        from packages.security.rbac import RBACEngine
        rbac = RBACEngine()
        rbac.assign_role("analyst", "OPERATOR")
        rbac.assign_role("commander", "MISSION_COMMANDER")
        assert rbac.get_max_classification("analyst") == "CONFIDENTIAL"
        assert rbac.get_max_classification("commander") == "SECRET"

    def test_data_classification_no_declassify(self):
        from packages.security.classification import (
            ClassificationPolicy, DataClassification, ClassificationLevel,
        )
        policy = ClassificationPolicy()
        policy.label("doc1", DataClassification(level=ClassificationLevel.SECRET))
        with pytest.raises(PermissionError):
            policy.label("doc1", DataClassification(level=ClassificationLevel.UNCLASSIFIED))

    def test_data_classification_upgrade(self):
        from packages.security.classification import (
            ClassificationPolicy, DataClassification, ClassificationLevel,
        )
        policy = ClassificationPolicy()
        policy.label("doc1", DataClassification(level=ClassificationLevel.CONFIDENTIAL))
        upgraded = policy.upgrade("doc1", ClassificationLevel.SECRET)
        assert upgraded.level == ClassificationLevel.SECRET

    def test_mtls_ca_init(self):
        from packages.security.mtls import CertificateAuthority
        ca = CertificateAuthority(org="TestOrg")
        cert = ca.init_ca()
        assert cert.common_name == "TestOrg Root CA"
        assert cert.is_ca
        assert not cert.is_expired


# ── Mesh Transport Tests ────────────────────────────────────

class TestMeshTransport:
    """Test mesh transport framing and encryption."""

    def test_frame_encode_decode(self):
        from packages.mesh.transport import FramedMessage, MessageType
        msg = FramedMessage(
            msg_type=MessageType.HEARTBEAT,
            payload=b'{"node_id": "test"}',
        )
        encoded = msg.encode()
        decoded = FramedMessage.decode(encoded)
        assert decoded is not None
        assert decoded.msg_type == MessageType.HEARTBEAT
        assert decoded.payload == b'{"node_id": "test"}'

    def test_frame_hmac_verification(self):
        from packages.mesh.transport import FramedMessage, MessageType
        key = b"secret_key_12345"
        msg = FramedMessage(msg_type=MessageType.DATA, payload=b"hello")
        encoded = msg.encode(hmac_key=key)
        # Verify with correct key
        decoded = FramedMessage.decode(encoded, hmac_key=key)
        assert decoded is not None
        # Verify with wrong key
        bad = FramedMessage.decode(encoded, hmac_key=b"wrong_key")
        assert bad is None

    def test_encryption_roundtrip(self):
        from packages.mesh.transport import EncryptionEnvelope
        import os
        key = os.urandom(32)
        env = EncryptionEnvelope(key)
        nonce = os.urandom(16)
        plaintext = b"classified data"
        ciphertext = env.encrypt(plaintext, nonce)
        assert ciphertext != plaintext
        recovered = env.decrypt(ciphertext, nonce)
        assert recovered == plaintext


# ── gRPC Service Tests ──────────────────────────────────────

class TestGRPCServices:
    """Test gRPC entity and task services."""

    @pytest.mark.asyncio
    async def test_entity_crud(self):
        from packages.grpc_services.entity_service import EntityServicer
        svc = EntityServicer()
        # Create
        resp = await svc.CreateEntity({
            "entity_type": "track", "domain": "AIR",
            "lat": 34.0, "lon": -118.0, "alt": 10000,
        })
        entity_id = resp["entity"]["entity_id"]
        assert entity_id

        # Read
        resp = await svc.GetEntity({"entity_id": entity_id})
        assert resp["entity"]["domain"] == "AIR"

        # Update
        resp = await svc.UpdateEntity({"entity_id": entity_id, "speed": 250.0})
        assert resp["entity"]["speed"] == 250.0

        # Delete
        resp = await svc.DeleteEntity({"entity_id": entity_id})
        assert resp["deleted"]

    @pytest.mark.asyncio
    async def test_entity_list_filter(self):
        from packages.grpc_services.entity_service import EntityServicer
        svc = EntityServicer()
        await svc.CreateEntity({"entity_type": "track", "domain": "AIR"})
        await svc.CreateEntity({"entity_type": "track", "domain": "GROUND"})
        await svc.CreateEntity({"entity_type": "track", "domain": "AIR"})

        resp = await svc.ListEntities({"domain": "AIR"})
        assert resp["total"] == 2

    @pytest.mark.asyncio
    async def test_task_lifecycle(self):
        from packages.grpc_services.task_service import TaskServicer
        svc = TaskServicer()
        # Create
        resp = await svc.CreateTask({
            "task_type": "navigate",
            "target_lat": 34.0, "target_lon": -118.0,
            "mission_id": "m1",
        })
        task_id = resp["task"]["task_id"]

        # Assign
        resp = await svc.AssignTask({"task_id": task_id, "assignee_id": "drone_1"})
        assert resp["task"]["state"] == "assigned"

        # Start
        resp = await svc.StartTask({"task_id": task_id})
        assert resp["task"]["state"] == "running"

        # Complete
        resp = await svc.CompleteTask({"task_id": task_id, "result": {"arrived": True}})
        assert resp["task"]["state"] == "completed"

    @pytest.mark.asyncio
    async def test_task_dependencies(self):
        from packages.grpc_services.task_service import TaskServicer
        svc = TaskServicer()
        r1 = await svc.CreateTask({"task_type": "recon", "task_id": "t1"})
        r2 = await svc.CreateTask({"task_type": "strike", "task_id": "t2", "depends_on": ["t1"]})

        # Can't assign t2 before t1 completes
        resp = await svc.AssignTask({"task_id": "t2", "assignee_id": "drone_1"})
        assert "error" in resp

        # Complete t1
        await svc.AssignTask({"task_id": "t1", "assignee_id": "drone_1"})
        await svc.StartTask({"task_id": "t1"})
        await svc.CompleteTask({"task_id": "t1"})

        # Now t2 can be assigned
        resp = await svc.AssignTask({"task_id": "t2", "assignee_id": "drone_2"})
        assert resp["task"]["state"] == "assigned"


# ── Fusion Depth Tests ──────────────────────────────────────

class TestFusionDepth:
    """Test UKF and MHT."""

    def test_ukf_converges(self):
        from apps.fusion.filters.ukf import UnscentedKalmanFilter
        ukf = UnscentedKalmanFilter()
        # Feed measurements moving north
        for i in range(10):
            ukf.update(34.0 + i * 0.001, -118.0, 1000.0, timestamp=float(i))
        lat, lon, alt = ukf.position
        assert lat > 34.005  # Should have tracked northward movement
        assert ukf.speed > 0  # Should estimate nonzero speed

    def test_mht_creates_tracks(self):
        from apps.fusion.multi_hypothesis import MultiHypothesisTracker, Measurement
        mht = MultiHypothesisTracker()
        measurements = [
            Measurement(idx=0, lat=34.0, lon=-118.0),
            Measurement(idx=1, lat=34.1, lon=-118.1),
        ]
        result = mht.process_scan(measurements)
        assert len(result["new_tracks"]) >= 1
        assert result["hypotheses"] > 0


# ── Tracing Tests ───────────────────────────────────────────

class TestTracing:
    """Test distributed tracing."""

    def test_span_lifecycle(self):
        from packages.observability.tracing import Tracer, InMemoryExporter
        exporter = InMemoryExporter()
        tracer = Tracer(service_name="test", exporter=exporter)

        with tracer.start_span("test_op") as span:
            span.set_attribute("key", "value")
            span.add_event("checkpoint")

        assert len(exporter.spans) == 1
        assert exporter.spans[0].operation == "test_op"
        assert exporter.spans[0].status == "ok"
        assert exporter.spans[0].duration_ms > 0

    def test_nested_spans(self):
        from packages.observability.tracing import Tracer, InMemoryExporter
        exporter = InMemoryExporter()
        tracer = Tracer(service_name="test", exporter=exporter)

        with tracer.start_span("parent") as parent:
            with tracer.start_span("child") as child:
                assert child.parent_id == parent.span_id
                assert child.trace_id == parent.trace_id

        assert len(exporter.spans) == 2
