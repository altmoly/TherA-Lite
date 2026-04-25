# TherA Lite

TherA Lite is a dark themed Next.js + TypeScript + Tailwind prototype for client-side thermal reasoning from RGB images. Users can upload an image, capture from a webcam, and generate a relative inferno-palette thermal estimate with object-level explanations.

## Features

- Image upload
- Webcam capture
- Client-side thermal prediction with TensorFlow.js
- COCO-SSD object detection for people, animals, vehicles, and common scene objects
- BlazeFace face detection for facial skin reasoning
- Optional BodyPix person segmentation when a person is detected
- Mask-based scene reasoning for sky, water, vegetation, sunlit road/concrete, and shadows
- Separate cue masks for faces, skin, people, hair, clothing, animals, vehicles, sky, water, vegetation, ground, walls, windows, glass/metal, lamps, sunlight, and shadows
- Expanded environmental priors for clouds, fog, wet surfaces, snow/ice, leaves, grass, trees, forest shade, soil, dry soil, sand, rocks, asphalt, concrete, brick, and roofs
- FLIR-like ironbow thermal heatmap overlay
- Optional debug masks mode for inspecting detected categories
- Optional human mode tuned for portraits
- Side-by-side original and thermal views
- AI reasoning panel listing detected objects and why regions were hot, warm, cool, or mixed
- Confidence score based on detector confidence, image quality, and segmentation availability
- Sliders for human heat boost, sunlight boost, and thermal contrast
- Static export configuration for Firebase Hosting

## Getting Started

Install dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Build

Create the static production export:

```bash
npm run build
```

Next.js writes the deployable static site to `out/`.

Preview the static export locally:

```bash
npm run preview
```

## Firebase Hosting

Install and log in to the Firebase CLI if needed:

```bash
npm install -g firebase-tools
firebase login
```

Set your Firebase project in `.firebaserc` or run:

```bash
firebase use --add
```

Deploy:

```bash
npm run build
firebase deploy
```

## Notes on Inference

TherA Lite does not call a server or external AI API. It loads TensorFlow.js models in the browser and combines object detections with canvas masks:

- COCO-SSD detections mark people and animals as likely hot.
- BlazeFace boosts facial skin regions, with forehead and cheeks warmer and the nose tip slightly cooler.
- Weak detections are ignored, so people and animals are not invented from random bright regions.
- Vehicles are modeled as warm only in likely engine, tire, brake, and exhaust areas instead of heating the full body panel.
- BodyPix person segmentation is used when available to make person masks cleaner than bounding boxes alone; fallback person masks are body-shaped, not rectangular.
- Skin-colored pixels are weighted higher only when they align with a detected face or person mask.
- Hair and clothes are cooler than skin in person masks.
- Blue upper-scene regions are cooled as likely sky.
- Blue lower-scene regions are cooled as likely water.
- Green saturated regions are treated as cool-medium vegetation.
- Road and concrete-like regions warm up only when bright or sunlit.
- Glass and metal-like regions are treated as reflective and are not automatically hot.
- Ceiling lights and lamps are neutralized so indoor lighting is not treated as heat.
- Low-brightness, low-saturation regions receive a shadow cooling adjustment.
- Shadow-aware local normalization reduces the influence of illumination on the estimate.
- Mask edges are smoothed only across visually similar neighboring pixels to reduce bleeding into nearby objects.
- All class masks are blended through relative thermal priors such as skin `0.95`, animals `0.85`, cars `0.70`, asphalt `0.60`, concrete `0.50`, trees `0.35`, water `0.20`, sky `0.10`, and snow/ice `0.02`.
- Small distant people, animals, and vehicles are preserved but cooled slightly to simulate depth and atmospheric loss.

DeepLab semantic segmentation was evaluated, but the published package currently conflicts with the TensorFlow.js 4.x stack used by COCO-SSD, BodyPix, and BlazeFace. TherA Lite therefore keeps the segmentation fully client-side by combining TensorFlow object/body/face models with explicit semantic material masks and priors.

This is not calibrated thermal imaging and does not measure real temperature. It is a visual reasoning prototype for product exploration and UI testing.
