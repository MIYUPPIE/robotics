# **Project: Automatic Environment-Aware Webcam Light Compensation System**



Selected engineering direction

Computer Vision

Python + OpenCV



We will develop this as an autonomous computer vision system that uses your laptop webcam to observe the room, learn its normal illumination level, detect when the lights go off, and automatically compensate for the reduction in brightness.

The objective is to make the **camera output retain approximately the same perceived brightness as the original illuminated environment**, even when the physical room becomes dark.

One important engineering distinction: we are preserving the visual appearance of the camera feed, not physically preserving the room's illumination. At extremely low light levels, the camera may not capture enough information to reconstruct details perfectly.

## **1. System requirements**

| Requirement         | Engineering decision                     |
| ------------------- | ---------------------------------------- |
| Camera              | Built-in laptop webcam                   |
| Reference           | Initial environmental illumination       |
| Operation           | Fully automatic                          |
| Light detection     | Frame luminance analysis                 |
| Reference retention | Persistent calibration                   |
| Compensation        | Adaptive gamma + CLAHE                   |
| Stability           | Temporal smoothing and hysteresis        |
| Output              | Real-time enhanced video                 |
| Logging             | Light intensity and compensation history |
| Platform            | Pop!\_OS / Linux                         |
| Language            | Python                                   |

### **Expected behaviour**

**Normal environment**

Reference state

[How to Set Up Perfect Lighting for Video Calls in 2026 | Home Desk Essentials](https://images.openai.com/static-rsc-4/PrXJkwUm1MnC7Leyy4TpzcetTaYyb66_b9RGTp-0tuktTsFNp4JiVI3ZXdimByMX4EXOdR0OhsrLBPuFJbnWvWsWQTRr8RyZVyPkh0WR25oF_mKjwcUUzULI-T5kyc8uPgFdlqmQbSAorSEpmzhFguFfloQuPtWGi1lgzMmrLUM?purpose=inline)

[homedeskessentials.com](https://www.homedeskessentials.com/blog/video-call-lighting-setup?utm_source=chatgpt.com)

The system learns the initial environmental brightness.

**Lights switched off**

Change detected

[Dell Pro Webcam (WB5023) Review: Perfect Plug-and-Play in 2K | Tom's Hardware](https://images.openai.com/static-rsc-4/notAiEGHSX02xJHqJ7zJV-gmaW2HpCO0y5_ZYCdmECKndyfL8hrCrRX0sg_1MaUYeaTVtLnrQHZe6Ehm6UTGhuWzUiMIm0PudZ1wqkK7iUNKm3iP4NcMU45Pp_knPJufm758eFRs6rFgxKsxpCP9USDHI3-1QMKxrcbVQawyEfs?purpose=inline)

[tomshardware.com](https://www.tomshardware.com/reviews/dell-pro-webcam-wb5023?utm_source=chatgpt.com)

The system detects a substantial drop in illumination.

**Compensated camera output**

Enhanced state

[Dell's New $200 Webcam Uses a Sony Low-Light CMOS Sensor | PetaPixel](https://images.openai.com/static-rsc-4/W01cdIntgJzQje_ifzXRMQSiQuXAA1WzbmsiXWN7M4gWAt-J4CNtb2FDy27HYX_CYcnMQy7hs9xzWrZOHCahJkPViHNK0Im43peofcNGw-vo-6DTrRQSIQPSROTE-gNs3cosmBqiT6bcknUZ1pXQ5YF-6Q8sGLwjdIRYpyFrtb0?purpose=inline)

[petapixel.com](https://petapixel.com/2021/06/30/dells-new-200-webcam-uses-a-sony-low-light-industrial-sensor/?utm_source=chatgpt.com)

Digital enhancement attempts to restore the original perceived brightness.

## **2. Advanced system architecture**

I recommend dividing the software into eight independent modules instead of putting all the processing inside one Python script.

**01. Acquisition layer**

Webcam capture, resolution, frame rate and exposure settings

**02. Image preprocessing**

Denoising, resizing, luminance-channel extraction

**03. Environmental illumination estimator**

Mean, median, percentiles and regional brightness

**04. Reference environment manager**

Calibration, reference storage and drift management

**05. Adaptive light-state detector**

Normal, dimmed, dark and recovery states

**06. Compensation controller**

Calculates the required brightness correction

**07. Image enhancement engine**

Gamma correction, CLAHE, denoising and colour preservation

**08. Output and monitoring**

Enhanced video, intensity graph, event log and performance metrics

### **2.1 The most important engineering decision: camera exposure**

The webcam has its own automatic exposure and gain mechanisms.

If the room becomes dark, the webcam may automatically increase exposure. This can make the image appear brighter without our software doing anything.

For a proper experiment, we should:

1. Capture the normal environment.
2. Calibrate the camera exposure.
3. Lock exposure and gain where the webcam supports it.
4. Measure changes in environmental illumination from subsequent frames.
5. Apply our own compensation algorithm.

This separates camera auto-exposure behaviour from our computer vision algorithm.

We will also use the median and percentile luminance rather than relying exclusively on average brightness. A person walking into the camera view should not cause the system to mistake a scene change for a light switch.

### **2.2 Mathematical model**

We define the reference illumination:

\\[ L\_{ref} = \operatorname{median}(L_1,L_2,\ldots,L_n) \\]

Here, \\(n\\) represents the calibration frames.

For every incoming frame:

\\[ R_t = \frac{L\_{ref}}{\max(L_t,\epsilon)} \\]

Where:

- \\(L\_{ref}\\) is the reference luminance.
- \\(L_t\\) is the current luminance.
- \\(R_t\\) is the compensation ratio.
- \\(\epsilon\\) prevents division by zero.

The controller uses a bounded correction:

\\[ G_t = \operatorname{clip}(R_t^\alpha,1,G\_{max}) \\]

We will smooth the controller response:

\\[ G_t^{smooth} = \beta G\_{t-1}^{smooth}+(1-\beta)G_t \\]

This prevents sudden brightness jumps and flickering.

The final image will use adaptive gamma and local contrast enhancement, rather than simply multiplying every pixel by the same brightness factor.

## **3. Development environment**

Let's build the first working version on your Pop!\_OS machine.

### **Step 1: Create the project**

```
mkdir -p ~/light-compensation/{src,config,data,logs,models,tests}
cd ~/light-compensation

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install opencv-python numpy matplotlib pandas
```

Check your camera:

```
ls /dev/video*
```

Install the V4L2 utilities if necessary:

```
sudo apt install v4l-utils
```

Inspect the webcam's supported controls:

```
v4l2-ctl --list-ctrls
```

Look for:

- `exposure_auto`
- `exposure_absolute`
- `gain`
- `brightness`
- `contrast`

Do not assume every control is supported by your webcam.

### **Step 2: Create the project structure**

```
light-compensation/
│
├── main.py
│
├── config/
│   └── settings.json
│
├── src/
│   ├── camera.py
│   ├── preprocessing.py
│   ├── illumination.py
│   ├── calibration.py
│   ├── light_detector.py
│   ├── controller.py
│   ├── enhancer.py
│   └── logger.py
│
├── data/
│   └── reference.json
│
├── logs/
│
├── models/
│
└── tests/
```

This separation will allow us to replace the initial image enhancement algorithm with a trained deep learning model later without rewriting the entire application.

## **4. Implement the first advanced prototype**

We will build Version 0.1 with:

- Automatic initial calibration.
- Reference illumination retention.
- Continuous environmental monitoring.
- Light-off detection.
- Hysteresis to avoid rapid state switching.
- Adaptive gamma correction.
- CLAHE local contrast enhancement.
- Original-versus-enhanced video.
- CSV logging.

For the first version, use a single executable file to validate the algorithm before distributing it across modules.

### **Step 3: Create&#x20;**`main.py`

```
import cv2import numpy as npimport timeimport csvimport osfrom collections import dequefrom datetime import datetime# ==========================================# CONFIGURATION# ==========================================CAMERA_INDEX = 0WIDTH = 640HEIGHT = 480CALIBRATION_SECONDS = 5DARK_THRESHOLD = 0.72RECOVERY_THRESHOLD = 0.84MAX_GAMMA_CORRECTION = 0.35EMA_ALPHA = 0.15LOG_FILE = "logs/light_history.csv"os.makedirs("logs", exist_ok=True)# ==========================================# CAMERA INITIALIZATION# ==========================================cap = cv2.VideoCapture(CAMERA_INDEX)cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
```

Run the application:

```
cd ~/light-compensation
source .venv/bin/activate
python main.py
```

Implementation note: this is the first experimental controller, not yet a production-grade illumination estimator. The gamma floor and CLAHE parameters will need tuning against real webcam recordings.

## **5. Camera exposure calibration**

Before judging the compensation algorithm, we need to understand your webcam's exposure behaviour.

Run:

```
v4l2-ctl --list-ctrls
```

If the webcam supports manual exposure, we can configure it through V4L2.

On many Linux webcams, the controls are:

- `exposure_auto`: selects automatic or manual exposure.
- `exposure_absolute`: controls exposure duration.
- `gain`: controls sensor amplification.

The exact values depend on the camera driver. We should inspect the output before changing them.

**Why this matters:** If exposure remains automatic, the webcam may compensate for the darkness itself, making our measured illumination unreliable.

## **6. Testing and validation**

We need to test the system under controlled lighting conditions.

| Experiment | Physical condition         | Expected result                     |
| ---------- | -------------------------- | ----------------------------------- |
| Test 1     | Normal room lighting       | Establish reference                 |
| Test 2     | Lights switched off        | Detect darkness                     |
| Test 3     | Lights switched on         | Restore normal state                |
| Test 4     | Lights dimmed              | Detect partial reduction            |
| Test 5     | Person walks across camera | Avoid false light detection         |
| Test 6     | Screen brightness changes  | Avoid unnecessary compensation      |
| Test 7     | Uneven lighting            | Enhance dark regions                |
| Test 8     | Extremely dark environment | Apply maximum correction gracefully |

### **Performance metrics**

We should record these measurements:

- Detection latency: how quickly the system identifies a lighting change.
- Compensation latency: how quickly it adjusts the image.
- Brightness error: difference between reference and enhanced output.
- FPS: processing speed.
- Temporal stability: whether the output flickers.
- Image quality: noise, colour accuracy and retained details.

For brightness retention, we can calculate:

\\[ E_L = \frac{|L\_{output}-L\_{ref}|}{L\_{ref}}\times100 \\]

Our initial engineering target should be a brightness error below 15% in moderately dark environments, subject to image-quality constraints.

## **7. Evolution into a more advanced computer vision system**

Once the initial version works, I recommend three development stages.

**Version 0.1: Classical CV**

Start here

- OpenCV
- Luminance estimation
- Gamma correction
- CLAHE
- Automatic detection

  **Version 0.2: Intelligent enhancement**
- Adaptive exposure modelling
- Spatial illumination estimation
- Noise reduction
- Local brightness compensation
- Automatic parameter optimization

  **Version 1.0: Deep learning enhancement**
- Zero-DCE / Zero-DCE++
- Learned illumination enhancement
- Real-time inference
- GPU acceleration where beneficial
- Quality comparison against classical CV

For Version 1.0, investigate Zero-DCE++. It uses a lightweight deep network to estimate image-enhancement curves without requiring paired low-light and normal-light training images.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://li-chongyi.github.io\&sz=32)

li-chongyi.github.io+1



Research:

- [Zero-DCE research and implementation](https://li-chongyi.github.io/Proj_Zero-DCE?utm_source=chatgpt.com)
- [Zero-DCE++ paper and project](https://li-chongyi.github.io/Proj_Zero-DCE%2B%2B.html?utm_source=chatgpt.com)

Your RTX 2060 6GB should be useful for experimenting with GPU-based enhancement models.

## **8. What the finished application should look like**

[NeuralBox by NeuralCam](https://images.openai.com/static-rsc-4/Fp734pHhYsog52-o1NOmyq9MXnG58pBv4zm0Rw4HnPVQTCaRwPLirfw8iw1NpVhR_qYjQSXxR62KbTDudubElhUrTWnPXCuhBdlj2e41zyOthBLvb0fRPEkk6Fn0O8uhwrishIcAPKcNgqi3tv34gqbxD98bmKW0TvIWqOeYT5E?purpose=inline)

[neural.cam](https://neural.cam/neuralbox/?utm_source=chatgpt.com)

The eventual dashboard should contain:

1. Original webcam feed.
2. Compensated webcam feed.
3. Reference illumination.
4. Current illumination.
5. Light-state indicator.
6. Compensation strength.
7. Real-time illumination graph.
8. FPS and processing latency.
9. Calibration and reset controls.

## **9. Development roadmap**

| Milestone | Deliverable                        |
| --------- | ---------------------------------- |
| M1        | Webcam acquisition and calibration |
| M2        | Illumination detection             |
| M3        | Automatic compensation             |
| M4        | Exposure control and stability     |
| M5        | Noise reduction and image quality  |
| M6        | Real-time dashboard                |
| M7        | Deep learning enhancement          |
| M8        | Benchmarking and documentation     |

**My recommendation:** Do not start with deep learning. First establish whether the classical computer vision approach can reliably detect environmental light changes and preserve the visual brightness. That gives us a measurable baseline against which to evaluate the neural network.

The next useful development step is to separate the prototype into the eight Python modules, add unit tests, and implement persistent reference calibration.

📷Review project development milestonesTap to set up

One final engineering consideration: because the webcam is the only sensor, our system estimates *image-based illumination*, not absolute physical illuminance in lux. Absolute lux measurement would require a calibrated light sensor.