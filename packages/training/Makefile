DATA_DIR   ?= /tmp/heli-training-data
MODELS_DIR := ../../packages/c2_intel/models
EPOCHS     ?= 50
DEVICE     ?= auto

.PHONY: all download-all train-all export clean help

all: download-all train-all export

# ── Downloads ───────────────────────────────────────────────────────────────

download-all: download-detection download-anomaly download-battery download-timing download-intent download-damage download-corrosion download-crowd

download-detection:
	python datasets/visdrone.py  --output $(DATA_DIR)/visdrone
	python datasets/dfire.py     --output $(DATA_DIR)/dfire
	python datasets/heridal.py   --output $(DATA_DIR)/heridal

download-anomaly:
	python datasets/smap_msl.py  --output $(DATA_DIR)/smap_msl

download-battery:
	python datasets/nasa_battery.py --output $(DATA_DIR)/battery

download-timing:
	python datasets/opensky.py   --output $(DATA_DIR)/opensky

download-intent:
	python datasets/intent.py    --output $(DATA_DIR)/intent

download-damage:
	python -c "from datasets.xbd import download; download()"

download-corrosion:
	python -c "from datasets.corrosion import download; download()"

download-crowd:
	python -c "from datasets.crowd import download; download()"

# ── Training ────────────────────────────────────────────────────────────────

train-all: train-detection train-anomaly train-battery train-timing train-intent train-damage train-corrosion train-crowd

train-detection:
	python train_detection.py \
	  --visdrone  $(DATA_DIR)/visdrone \
	  --dfire     $(DATA_DIR)/dfire \
	  --heridal   $(DATA_DIR)/heridal \
	  --output    $(DATA_DIR)/models \
	  --epochs    $(EPOCHS) \
	  --device    $(DEVICE)

train-anomaly:
	python train_anomaly.py \
	  --data    $(DATA_DIR)/smap_msl \
	  --output  $(MODELS_DIR)

train-battery:
	python train_battery.py \
	  --data    $(DATA_DIR)/battery \
	  --output  $(MODELS_DIR)

train-timing:
	python train_timing.py \
	  --data    $(DATA_DIR)/opensky \
	  --output  $(MODELS_DIR)

train-intent:
	python train_intent.py \
	  --data    $(DATA_DIR)/intent \
	  --output  $(MODELS_DIR)

train-damage:
	python train_damage.py

train-corrosion:
	python train_corrosion.py

train-crowd:
	python train_crowd.py

# ── Export ──────────────────────────────────────────────────────────────────

export:
	python export.py --models-dir $(DATA_DIR)/models --c2-models-dir $(MODELS_DIR)

# ── Housekeeping ─────────────────────────────────────────────────────────────

clean:
	rm -rf $(DATA_DIR)

help:
	@echo "Heli.OS Training Pipeline"
	@echo ""
	@echo "Targets:"
	@echo "  make download-all        Download all public datasets"
	@echo "  make train-all           Train all models"
	@echo "  make export              Copy trained models to integration paths"
	@echo "  make all                 Full pipeline: download → train → export"
	@echo ""
	@echo "Options:"
	@echo "  DATA_DIR=/path           Training data directory (default: /tmp/heli-training-data)"
	@echo "  EPOCHS=50                YOLOv8 training epochs"
	@echo "  DEVICE=auto              Training device: auto, cpu, 0 (GPU index)"
	@echo ""
	@echo "Individual:"
	@echo "  make train-detection     YOLOv8 aerial detection (VisDrone+D-Fire+HERIDAL)"
	@echo "  make train-anomaly       Telemetry anomaly detector (SMAP/MSL)"
	@echo "  make train-battery       Battery degradation predictor (NASA)"
	@echo "  make train-timing        Action timing predictor (OpenSky)"
	@echo "  make train-intent        Mission intent classifier (FEMA/NIMS)"
	@echo "  make train-damage        Building damage classifier (xBD)"
	@echo "  make train-corrosion     Structural defect classifier (CODEBRIM)"
	@echo "  make train-crowd         Crowd density estimator (ShanghaiTech)"
