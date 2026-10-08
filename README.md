# Roadside Assistant for Hidden Pedestrians and Cyclists

A roadside camera (RSU) in CARLA detects pedestrians and cyclists hidden behind a parked vehicle
and warns an approaching car over MQTT. The car fuses the warning with its own camera and slows down
or brakes. Three occlusion scenarios are run with and without the warning.

## Contents
| File | Description |
|---|---|
| `roadsideAssistant_mqtt.ipynb` | Full system: RSU detection, MQTT messaging, vehicle policy, scenarios, metrics |
| `yolov8n.pt` | YOLOv8n weights (COCO) used by both cameras |
| `requirements.txt` | Python dependencies |
| `docs/Roadside_Assistant_Report.pdf` | Project report |
| `docs/Seeing-around-the-corner.pptx` | Presentation |

## Setup
Tested with Python 3.12 and the CARLA 0.9.16 server on Windows.

```bash
pip install -r requirements.txt
```
The `carla` package must match the CARLA server version (for a 0.9.15 server, install `carla==0.9.15`).

## Running
1. Start the CARLA server (`CarlaUE4.exe`).
2. In a separate terminal, start the MQTT broker: `amqtt`.
3. Open `roadsideAssistant_mqtt.ipynb` and run the cells in order.
   Cell 1 prints the client and server versions; Cell 8 should print `[MQTT] link mode: mqtt`.

Results are written to `results.csv`, `comparison.csv`, `log_<scenario>_<mode>.csv` and `speed_profile_<scenario>.png`.

## Results
| Scenario | Without RSU | With RSU |
|---|---|---|
| S1 pedestrian behind a van | near miss (1.58 m) | safe (5.32 m) |
| S2 cyclist behind a truck | collision | safe (7.60 m) |
| S3 15 pedestrians, heavy rain | near miss (0.76 m) | safe (2.76 m) |

## Authors
Dias Katrenov, Danial Khayatian, Shkelzen Tafili
