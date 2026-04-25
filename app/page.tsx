"use client";

import { ChangeEvent, useEffect, useRef, useState, type ReactNode } from "react";

type Settings = {
  humanHeatBoost: number;
  sunlightBoost: number;
  thermalContrast: number;
  humanMode: boolean;
};

type RegionStats = {
  face: number;
  skin: number;
  person: number;
  hair: number;
  fabric: number;
  animal: number;
  vehicle: number;
  vehicleHotspot: number;
  sky: number;
  cloud: number;
  fog: number;
  water: number;
  wetSurface: number;
  snowIce: number;
  vegetation: number;
  leaves: number;
  grass: number;
  tree: number;
  flower: number;
  forestShade: number;
  soil: number;
  drySoil: number;
  sand: number;
  rock: number;
  asphalt: number;
  concrete: number;
  brick: number;
  ground: number;
  wall: number;
  roof: number;
  window: number;
  glassMetal: number;
  lamp: number;
  sunlitSurface: number;
  shadow: number;
};

type MaskKey = keyof RegionStats;

type Detection = {
  bbox: [number, number, number, number];
  class: string;
  score: number;
};

type FaceDetection = {
  bbox: [number, number, number, number];
  score: number;
};

type Reason = {
  label: string;
  confidence: number;
  thermal: "hot" | "warm" | "cool" | "mixed";
  reason: string;
};

type ThermalResult = {
  url: string;
  debugUrl: string;
  stats: RegionStats;
  avgHeat: number;
  pixelCount: number;
  detections: Detection[];
  faces: FaceDetection[];
  reasons: Reason[];
  confidence: number;
  imageQuality: number;
  usedBodySegmentation: boolean;
  lowConfidence: boolean;
};

type LoadedModels = {
  detector: {
    detect: (image: HTMLCanvasElement | HTMLImageElement | HTMLVideoElement) => Promise<Detection[]>;
  } | null;
  bodyPix: {
    segmentPerson: (
      image: HTMLCanvasElement,
      options?: Record<string, unknown>
    ) => Promise<{ data: Uint8Array | Int32Array; width: number; height: number }>;
  } | null;
  faceDetector: {
    estimateFaces: (image: HTMLCanvasElement) => Promise<FaceDetection[]>;
  } | null;
};

type PixelFeatures = {
  brightness: number;
  saturation: number;
  hue: number;
  sky: boolean;
  cloud: boolean;
  fog: boolean;
  water: boolean;
  wetSurface: boolean;
  snowIce: boolean;
  vegetation: boolean;
  leaves: boolean;
  grass: boolean;
  tree: boolean;
  flower: boolean;
  forestShade: boolean;
  soil: boolean;
  drySoil: boolean;
  sand: boolean;
  rock: boolean;
  asphalt: boolean;
  concrete: boolean;
  brick: boolean;
  shadow: boolean;
  sunlitSurface: boolean;
  ground: boolean;
  wall: boolean;
  roof: boolean;
  window: boolean;
  glassMetal: boolean;
  lamp: boolean;
  skin: boolean;
  hair: boolean;
};

type Masks = Record<MaskKey, Uint8Array>;

const defaultSettings: Settings = {
  humanHeatBoost: 42,
  sunlightBoost: 28,
  thermalContrast: 62,
  humanMode: false
};

const animalObjects = new Set(["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]);
const vehicleObjects = new Set(["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]);

const maskColors: Record<MaskKey, [number, number, number]> = {
  face: [255, 129, 93],
  skin: [251, 113, 133],
  person: [244, 63, 94],
  hair: [30, 41, 59],
  fabric: [139, 92, 246],
  animal: [251, 146, 60],
  vehicle: [251, 191, 36],
  vehicleHotspot: [255, 245, 157],
  sky: [34, 211, 238],
  cloud: [203, 213, 225],
  fog: [148, 163, 184],
  water: [96, 165, 250],
  wetSurface: [59, 130, 246],
  snowIce: [224, 242, 254],
  vegetation: [52, 211, 153],
  leaves: [74, 222, 128],
  grass: [34, 197, 94],
  tree: [22, 163, 74],
  flower: [244, 114, 182],
  forestShade: [21, 128, 61],
  soil: [146, 100, 60],
  drySoil: [180, 112, 52],
  sand: [250, 204, 21],
  rock: [120, 113, 108],
  asphalt: [82, 82, 91],
  concrete: [161, 161, 170],
  brick: [185, 80, 54],
  ground: [180, 140, 92],
  wall: [168, 162, 158],
  roof: [202, 138, 74],
  window: [125, 211, 252],
  glassMetal: [192, 224, 255],
  lamp: [250, 250, 210],
  sunlitSurface: [253, 224, 71],
  shadow: [71, 85, 105]
};

function clamp(value: number, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function ironbow(t: number): [number, number, number] {
  const stops: Array<[number, number, number, number]> = [
    [0, 3, 6, 18],
    [0.14, 12, 20, 60],
    [0.3, 44, 24, 92],
    [0.46, 124, 28, 90],
    [0.61, 204, 55, 44],
    [0.76, 245, 132, 32],
    [0.9, 255, 214, 92],
    [1, 255, 255, 238]
  ];
  const x = clamp(t);
  const hi = stops.findIndex((stop) => stop[0] >= x);
  if (hi <= 0) return [stops[0][1], stops[0][2], stops[0][3]];
  const a = stops[hi - 1];
  const b = stops[hi];
  const k = (x - a[0]) / (b[0] - a[0]);
  return [
    Math.round(a[1] + (b[1] - a[1]) * k),
    Math.round(a[2] + (b[2] - a[2]) * k),
    Math.round(a[3] + (b[3] - a[3]) * k)
  ];
}

function rgbToHsv(r: number, g: number, b: number) {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const delta = max - min;
  let hue = 0;
  if (delta !== 0) {
    if (max === rn) hue = ((gn - bn) / delta) % 6;
    else if (max === gn) hue = (bn - rn) / delta + 2;
    else hue = (rn - gn) / delta + 4;
    hue *= 60;
    if (hue < 0) hue += 360;
  }
  return { hue, saturation: max === 0 ? 0 : delta / max, brightness: max };
}

function isSkinColor(r: number, g: number, b: number, saturation: number, brightness: number, hue: number) {
  const rgbSkin = r > 70 && g > 38 && b > 24 && r > b * 1.12 && r >= g * 0.9 && r - b > 18;
  const hsvSkin = (hue <= 50 || hue >= 340) && saturation > 0.16 && saturation < 0.72 && brightness > 0.22 && brightness < 0.96;
  return rgbSkin && hsvSkin;
}

function pixelFeatures(r: number, g: number, b: number, x: number, y: number, width: number, height: number): PixelFeatures {
  const { hue, saturation, brightness } = rgbToHsv(r, g, b);
  const upper = y < height * 0.48;
  const lower = y > height * 0.42;
  const middle = y > height * 0.18 && y < height * 0.82;
  const blue = hue >= 178 && hue <= 245;
  const green = hue >= 72 && hue <= 165;
  const neutral = saturation < 0.22;
  const specular = brightness > 0.78 && saturation < 0.18;
  const veryBrightNeutral = brightness > 0.86 && saturation < 0.2;
  const skin = isSkinColor(r, g, b, saturation, brightness, hue);
  const hair = brightness < 0.24 && saturation < 0.55 && !blue && !green;
  const redOrange = hue >= 0 && hue <= 34;
  const yellow = hue > 34 && hue <= 62;
  const magenta = hue >= 285 && hue <= 340;

  const sky = upper && blue && brightness > 0.34 && saturation > 0.12;
  const cloud = upper && !sky && saturation < 0.22 && brightness > 0.58 && brightness < 0.94;
  const fog = upper && !sky && saturation < 0.16 && brightness > 0.38 && brightness <= 0.7;
  const water = !sky && lower && blue && brightness > 0.18 && brightness < 0.76 && saturation > 0.1;
  const wetSurface = lower && !water && neutral && brightness > 0.2 && brightness < 0.48 && b >= r * 0.86;
  const snowIce = saturation < 0.18 && brightness > 0.78 && (upper || lower) && b >= r * 0.82;
  const vegetation = green && saturation > 0.2 && brightness > 0.12;
  const grass = vegetation && lower && brightness > 0.22;
  const tree = vegetation && !grass && y < height * 0.82;
  const leaves = vegetation && saturation > 0.32;
  const flower = (magenta || redOrange || yellow) && saturation > 0.36 && brightness > 0.28 && vegetation;
  const forestShade = vegetation && brightness < 0.28;
  const shadow = brightness < 0.22 && saturation < 0.5;
  const soil = lower && redOrange && saturation > 0.18 && saturation < 0.62 && brightness > 0.18 && brightness < 0.58;
  const drySoil = soil && brightness > 0.36 && saturation > 0.28;
  const sand = lower && yellow && saturation > 0.18 && brightness > 0.45;
  const rock = lower && neutral && brightness > 0.24 && brightness < 0.62;
  const asphalt = lower && neutral && brightness > 0.16 && brightness < 0.42;
  const concrete = lower && neutral && brightness >= 0.42 && brightness < 0.76;
  const brick = redOrange && saturation > 0.22 && brightness > 0.24 && brightness < 0.66 && !skin;
  const ground = lower && !sky && !water && !vegetation && (neutral || soil || sand || rock || asphalt || concrete) && brightness > 0.16 && brightness < 0.78;
  const wall = middle && !sky && !water && !vegetation && neutral && brightness >= 0.26 && brightness < 0.86 && x > width * 0.03;
  const roof = y < height * 0.58 && y > height * 0.18 && !sky && !water && !vegetation && (redOrange || neutral) && brightness > 0.22 && brightness < 0.78;
  const window = !sky && !water && blue && saturation < 0.34 && brightness > 0.28 && brightness < 0.88 && y < height * 0.85;
  const glassMetal =
    !vegetation &&
    !sky &&
    !water &&
    ((neutral && brightness > 0.46 && middle) || specular || (blue && saturation < 0.26 && brightness > 0.42));
  const lamp = upper && veryBrightNeutral && !sky && !water && !vegetation;
  const sunlitSurface = !sky && !water && !vegetation && !glassMetal && !lamp && brightness > 0.66 && y > height * 0.24;

  return {
    brightness,
    saturation,
    hue,
    sky,
    cloud,
    fog,
    water,
    wetSurface,
    snowIce,
    vegetation,
    leaves,
    grass,
    tree,
    flower,
    forestShade,
    soil,
    drySoil,
    sand,
    rock,
    asphalt,
    concrete,
    brick,
    shadow,
    sunlitSurface,
    ground,
    wall,
    roof,
    window,
    glassMetal,
    lamp,
    skin,
    hair
  };
}

function createMasks(pixelCount: number): Masks {
  return {
    face: new Uint8Array(pixelCount),
    skin: new Uint8Array(pixelCount),
    person: new Uint8Array(pixelCount),
    hair: new Uint8Array(pixelCount),
    fabric: new Uint8Array(pixelCount),
    animal: new Uint8Array(pixelCount),
    vehicle: new Uint8Array(pixelCount),
    vehicleHotspot: new Uint8Array(pixelCount),
    sky: new Uint8Array(pixelCount),
    cloud: new Uint8Array(pixelCount),
    fog: new Uint8Array(pixelCount),
    water: new Uint8Array(pixelCount),
    wetSurface: new Uint8Array(pixelCount),
    snowIce: new Uint8Array(pixelCount),
    vegetation: new Uint8Array(pixelCount),
    leaves: new Uint8Array(pixelCount),
    grass: new Uint8Array(pixelCount),
    tree: new Uint8Array(pixelCount),
    flower: new Uint8Array(pixelCount),
    forestShade: new Uint8Array(pixelCount),
    soil: new Uint8Array(pixelCount),
    drySoil: new Uint8Array(pixelCount),
    sand: new Uint8Array(pixelCount),
    rock: new Uint8Array(pixelCount),
    asphalt: new Uint8Array(pixelCount),
    concrete: new Uint8Array(pixelCount),
    brick: new Uint8Array(pixelCount),
    ground: new Uint8Array(pixelCount),
    wall: new Uint8Array(pixelCount),
    roof: new Uint8Array(pixelCount),
    window: new Uint8Array(pixelCount),
    glassMetal: new Uint8Array(pixelCount),
    lamp: new Uint8Array(pixelCount),
    sunlitSurface: new Uint8Array(pixelCount),
    shadow: new Uint8Array(pixelCount)
  };
}

function detectionArea(detection: Detection) {
  return detection.bbox[2] * detection.bbox[3];
}

function isHumanLikeDetection(detection: Detection) {
  const [, , w, h] = detection.bbox;
  const aspect = h / Math.max(w, 0.001);
  return detection.class === "person" && detection.score >= 0.58 && aspect >= 1.05 && aspect <= 4.8 && detectionArea(detection) > 0.006;
}

function isStrongAnimalDetection(detection: Detection) {
  return animalObjects.has(detection.class) && detection.score >= 0.52 && detectionArea(detection) > 0.004;
}

function isStrongVehicleDetection(detection: Detection) {
  const [, , w, h] = detection.bbox;
  return vehicleObjects.has(detection.class) && detection.score >= 0.5 && detectionArea(detection) > 0.008 && w > 0.04 && h > 0.035;
}

function normalizeDetections(raw: Detection[], width: number, height: number) {
  return raw
    .map((item) => ({
      ...item,
      bbox: [item.bbox[0] / width, item.bbox[1] / height, item.bbox[2] / width, item.bbox[3] / height] as [
        number,
        number,
        number,
        number
      ]
    }))
    .filter((item) => isHumanLikeDetection(item) || isStrongAnimalDetection(item) || isStrongVehicleDetection(item));
}

function colorDistance(data: Uint8ClampedArray, a: number, b: number) {
  const dr = data[a] - data[b];
  const dg = data[a + 1] - data[b + 1];
  const db = data[a + 2] - data[b + 2];
  return Math.sqrt(dr * dr + dg * dg + db * db) / 441.7;
}

function smoothMaskEdgeAware(mask: Uint8Array, data: Uint8ClampedArray, width: number, height: number, passes = 1) {
  let current = mask;
  for (let pass = 0; pass < passes; pass += 1) {
    const next = new Uint8Array(current);
    for (let y = 1; y < height - 1; y += 1) {
      for (let x = 1; x < width - 1; x += 1) {
        const p = y * width + x;
        const i = p * 4;
        let votes = 0;
        let eligible = 0;
        for (let oy = -1; oy <= 1; oy += 1) {
          for (let ox = -1; ox <= 1; ox += 1) {
            if (ox === 0 && oy === 0) continue;
            const np = (y + oy) * width + x + ox;
            const ni = np * 4;
            if (colorDistance(data, i, ni) > 0.16) continue;
            eligible += 1;
            votes += current[np];
          }
        }
        if (current[p] && votes <= 1) next[p] = 0;
        else if (!current[p] && eligible >= 5 && votes >= 6) next[p] = 1;
      }
    }
    current = next;
  }
  mask.set(current);
}

function applyEllipseMask(mask: Uint8Array, detection: Detection, width: number, height: number, data: Uint8ClampedArray, minStrength = 0.1) {
  const [nx, ny, nw, nh] = detection.bbox;
  const x0 = Math.max(0, Math.floor(nx * width));
  const y0 = Math.max(0, Math.floor(ny * height));
  const x1 = Math.min(width - 1, Math.ceil((nx + nw) * width));
  const y1 = Math.min(height - 1, Math.ceil((ny + nh) * height));
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  const rx = Math.max((x1 - x0) / 2, 1);
  const ry = Math.max((y1 - y0) / 2, 1);

  for (let y = y0; y <= y1; y += 1) {
    for (let x = x0; x <= x1; x += 1) {
      const px = (x - cx) / rx;
      const py = (y - cy) / ry;
      const strength = 1 - (px * px * 0.9 + py * py * 0.72);
      if (strength < minStrength) continue;
      const i = (y * width + x) * 4;
      const features = pixelFeatures(data[i], data[i + 1], data[i + 2], x, y, width, height);
      if (features.sky || features.water || features.vegetation) continue;
      mask[y * width + x] = 1;
    }
  }
}

function applyVehicleMask(mask: Uint8Array, detection: Detection, width: number, height: number, data: Uint8ClampedArray) {
  const [nx, ny, nw, nh] = detection.bbox;
  const x0 = Math.max(0, Math.floor(nx * width));
  const y0 = Math.max(0, Math.floor(ny * height));
  const x1 = Math.min(width - 1, Math.ceil((nx + nw) * width));
  const y1 = Math.min(height - 1, Math.ceil((ny + nh) * height));
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  const rx = Math.max((x1 - x0) / 2, 1);
  const ry = Math.max((y1 - y0) / 2, 1);

  for (let y = y0; y <= y1; y += 1) {
    for (let x = x0; x <= x1; x += 1) {
      const px = Math.abs((x - cx) / rx);
      const py = Math.abs((y - cy) / ry);
      if (px > 1 || py > 1) continue;
      const i = (y * width + x) * 4;
      const features = pixelFeatures(data[i], data[i + 1], data[i + 2], x, y, width, height);
      if (features.sky || features.water || features.vegetation) continue;
      if (py < 0.92 && px < 0.98) mask[y * width + x] = 1;
    }
  }
}

function vehicleHotspot(x: number, y: number, detection: Detection, width: number, height: number) {
  const [nx, ny, nw, nh] = detection.bbox;
  const x0 = nx * width;
  const y0 = ny * height;
  const w = nw * width;
  const h = nh * height;
  const lx = (x - x0) / Math.max(w, 1);
  const ly = (y - y0) / Math.max(h, 1);
  const front = Math.exp(-((lx - 0.22) ** 2 / 0.035 + (ly - 0.58) ** 2 / 0.12));
  const rear = Math.exp(-((lx - 0.82) ** 2 / 0.04 + (ly - 0.65) ** 2 / 0.13));
  const tireA = Math.exp(-((lx - 0.25) ** 2 / 0.015 + (ly - 0.88) ** 2 / 0.018));
  const tireB = Math.exp(-((lx - 0.75) ** 2 / 0.015 + (ly - 0.88) ** 2 / 0.018));
  return clamp(Math.max(front, rear * 0.85, tireA, tireB));
}

function depthCoolingFromDetection(detection: Detection) {
  const area = detectionArea(detection);
  const [, y, , h] = detection.bbox;
  const bottom = y + h;
  const sizeCooling = area < 0.02 ? 0.08 : area < 0.05 ? 0.04 : 0;
  const horizonCooling = bottom < 0.55 ? 0.06 : bottom < 0.72 ? 0.03 : 0;
  return sizeCooling + horizonCooling;
}

const thermalPriors: Record<MaskKey, number> = {
  face: 0.95,
  skin: 0.9,
  person: 0.72,
  hair: 0.42,
  fabric: 0.45,
  animal: 0.85,
  vehicle: 0.62,
  vehicleHotspot: 0.78,
  sky: 0.1,
  cloud: 0.22,
  fog: 0.18,
  water: 0.2,
  wetSurface: 0.26,
  snowIce: 0.02,
  vegetation: 0.35,
  leaves: 0.32,
  grass: 0.3,
  tree: 0.36,
  flower: 0.38,
  forestShade: 0.24,
  soil: 0.46,
  drySoil: 0.58,
  sand: 0.68,
  rock: 0.52,
  asphalt: 0.6,
  concrete: 0.5,
  brick: 0.48,
  ground: 0.45,
  wall: 0.42,
  roof: 0.62,
  window: 0.24,
  glassMetal: 0.36,
  lamp: 0.4,
  sunlitSurface: 0.56,
  shadow: 0.22
};

const semanticPriority: MaskKey[] = [
  "face",
  "skin",
  "animal",
  "vehicleHotspot",
  "vehicle",
  "person",
  "hair",
  "fabric",
  "snowIce",
  "sky",
  "cloud",
  "fog",
  "water",
  "wetSurface",
  "forestShade",
  "flower",
  "leaves",
  "grass",
  "tree",
  "vegetation",
  "sand",
  "drySoil",
  "soil",
  "asphalt",
  "concrete",
  "brick",
  "rock",
  "roof",
  "window",
  "glassMetal",
  "lamp",
  "wall",
  "ground",
  "shadow",
  "sunlitSurface"
];

function insideFace(face: FaceDetection, x: number, y: number, width: number, height: number) {
  const [nx, ny, nw, nh] = face.bbox;
  const fx = nx * width;
  const fy = ny * height;
  const fw = nw * width;
  const fh = nh * height;
  if (x < fx || x > fx + fw || y < fy || y > fy + fh) return null;
  return {
    localX: (x - fx) / Math.max(fw, 1),
    localY: (y - fy) / Math.max(fh, 1)
  };
}

function faceZoneHeat(face: FaceDetection, x: number, y: number, width: number, height: number) {
  const local = insideFace(face, x, y, width, height);
  if (!local) return 0;
  const dx = Math.abs(local.localX - 0.5);
  const dy = Math.abs(local.localY - 0.5);
  const oval = 1 - (dx * dx / 0.22 + dy * dy / 0.32);
  if (oval <= 0) return 0;

  const forehead = local.localY > 0.16 && local.localY < 0.38 && dx < 0.26 ? 0.82 : 0;
  const cheeks = local.localY > 0.42 && local.localY < 0.68 && dx > 0.16 && dx < 0.38 ? 0.8 : 0;
  const nose = local.localY > 0.42 && local.localY < 0.72 && dx < 0.12 ? 0.62 : 0;
  return Math.max(0.68 * oval, forehead, cheeks, nose);
}

function buildIlluminationMap(data: ImageData, width: number, height: number) {
  const brightness = new Float32Array(width * height);
  const local = new Float32Array(width * height);
  for (let p = 0; p < width * height; p += 1) {
    const i = p * 4;
    brightness[p] = (data.data[i] + data.data[i + 1] + data.data[i + 2]) / 765;
  }
  const radius = Math.max(8, Math.round(Math.min(width, height) / 34));
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let sum = 0;
      let count = 0;
      for (let oy = -radius; oy <= radius; oy += 4) {
        const yy = y + oy;
        if (yy < 0 || yy >= height) continue;
        for (let ox = -radius; ox <= radius; ox += 4) {
          const xx = x + ox;
          if (xx < 0 || xx >= width) continue;
          sum += brightness[yy * width + xx];
          count += 1;
        }
      }
      local[y * width + x] = sum / Math.max(count, 1);
    }
  }
  return { brightness, local };
}

async function imageFromUrl(url: string) {
  const image = new Image();
  image.src = url;
  await image.decode();
  return image;
}

async function loadModels(): Promise<LoadedModels> {
  const tf = await import("@tensorflow/tfjs");
  await tf.ready();
  const coco = await import("@tensorflow-models/coco-ssd");
  const cocoModel = await coco.load({ base: "lite_mobilenet_v2" });
  const detector: LoadedModels["detector"] = {
    detect: async (image) => {
      const detections = await cocoModel.detect(image);
      return detections.map((item) => ({
        class: item.class,
        score: item.score,
        bbox: [item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3]]
      }));
    }
  };

  let bodyPix: LoadedModels["bodyPix"] = null;
  let faceDetector: LoadedModels["faceDetector"] = null;
  try {
    const bodyPixModule = await import("@tensorflow-models/body-pix");
    const bodyPixModel = await bodyPixModule.load({
      architecture: "MobileNetV1",
      outputStride: 16,
      multiplier: 0.75,
      quantBytes: 2
    });
    bodyPix = {
      segmentPerson: async (canvas, options) =>
        bodyPixModel.segmentPerson(canvas, options as Parameters<typeof bodyPixModel.segmentPerson>[1])
    };
  } catch {
    bodyPix = null;
  }

  try {
    const blazeface = await import("@tensorflow-models/blazeface");
    const faceModel = await blazeface.load();
    faceDetector = {
      estimateFaces: async (canvas) => {
        const predictions = await faceModel.estimateFaces(canvas, false);
        return predictions
          .map((prediction) => {
            const topLeft = prediction.topLeft as [number, number];
            const bottomRight = prediction.bottomRight as [number, number];
            const probability = Array.isArray(prediction.probability)
              ? Number(prediction.probability[0])
              : Number(prediction.probability ?? 0);
            return {
              bbox: [
                topLeft[0] / canvas.width,
                topLeft[1] / canvas.height,
                (bottomRight[0] - topLeft[0]) / canvas.width,
                (bottomRight[1] - topLeft[1]) / canvas.height
              ] as [number, number, number, number],
              score: probability
            };
          })
          .filter((face) => face.score >= 0.78 && face.bbox[2] * face.bbox[3] > 0.006);
      }
    };
  } catch {
    faceDetector = null;
  }

  return { detector, bodyPix, faceDetector };
}

function imageQualityScore(data: ImageData) {
  let brightness = 0;
  let contrast = 0;
  const step = 16;
  const samples: number[] = [];
  for (let i = 0; i < data.data.length; i += 4 * step) {
    const value = (data.data[i] + data.data[i + 1] + data.data[i + 2]) / 765;
    samples.push(value);
    brightness += value;
  }
  const mean = brightness / Math.max(samples.length, 1);
  for (const value of samples) contrast += Math.abs(value - mean);
  const exposure = 1 - Math.min(1, Math.abs(mean - 0.48) * 1.9);
  const detail = Math.min(1, (contrast / Math.max(samples.length, 1)) * 5.5);
  return clamp(exposure * 0.48 + detail * 0.52);
}

function countMasks(masks: Masks): RegionStats {
  const stats = {} as RegionStats;
  (Object.keys(masks) as MaskKey[]).forEach((key) => {
    let count = 0;
    const mask = masks[key];
    for (let i = 0; i < mask.length; i += 1) count += mask[i];
    stats[key] = count;
  });
  return stats;
}

function dominantMask(masks: Masks, p: number) {
  return semanticPriority.find((key) => masks[key][p]) ?? null;
}

function buildDebugMaskUrl(masks: Masks, width: number, height: number) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  const image = ctx.createImageData(width, height);
  const keys: MaskKey[] = [
    "sky",
    "cloud",
    "fog",
    "water",
    "wetSurface",
    "snowIce",
    "vegetation",
    "leaves",
    "grass",
    "tree",
    "flower",
    "forestShade",
    "soil",
    "drySoil",
    "sand",
    "rock",
    "asphalt",
    "concrete",
    "brick",
    "ground",
    "wall",
    "roof",
    "window",
    "glassMetal",
    "lamp",
    "sunlitSurface",
    "shadow",
    "vehicle",
    "vehicleHotspot",
    "animal",
    "fabric",
    "hair",
    "skin",
    "face",
    "person"
  ];
  for (let p = 0; p < width * height; p += 1) {
    let color: [number, number, number] = [12, 18, 32];
    for (const key of keys) {
      if (masks[key][p]) color = maskColors[key];
    }
    const i = p * 4;
    image.data[i] = color[0];
    image.data[i + 1] = color[1];
    image.data[i + 2] = color[2];
    image.data[i + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  return canvas.toDataURL("image/png");
}

function buildReasons(
  detections: Detection[],
  faces: FaceDetection[],
  stats: RegionStats,
  total: number,
  lowConfidence: boolean,
  humanMode: boolean
): Reason[] {
  const reasons: Reason[] = [];
  if (lowConfidence) {
    reasons.push({
      label: "low confidence scene estimate",
      confidence: 0.42,
      thermal: "mixed",
      reason: "Object detections were weak or image quality was limited, so the app used conservative material masks instead of inventing people or animals."
    });
  }

  if (humanMode) {
    reasons.push({
      label: "human portrait mode",
      confidence: faces.length > 0 || stats.person / total > 0.04 ? 0.76 : 0.48,
      thermal: "mixed",
      reason: "Portrait tuning is active: forehead and cheeks are warmed, nose tip is slightly cooler, hair and clothes are cooled relative to skin."
    });
  }

  for (const face of faces) {
    reasons.push({
      label: "face",
      confidence: face.score,
      thermal: "hot",
      reason: "A confident face detection was found, so facial skin is boosted more than nearby warm backgrounds."
    });
  }

  for (const detection of detections) {
    if (isHumanLikeDetection(detection)) {
      reasons.push({
        label: "person",
        confidence: detection.score,
        thermal: "hot",
        reason: "A confident human-shaped person detection was found; heat is applied only inside the body-shaped mask."
      });
    } else if (isStrongAnimalDetection(detection)) {
      reasons.push({
        label: detection.class,
        confidence: detection.score,
        thermal: "hot",
        reason: "A confident animal detection was found; animal heat is limited to the refined object mask."
      });
    } else if (isStrongVehicleDetection(detection)) {
      reasons.push({
        label: detection.class,
        confidence: detection.score,
        thermal: "warm",
        reason: "A confident vehicle detection was found; heat is concentrated near likely engine, tire, brake, and exhaust areas."
      });
    }
  }

  const materialReasons: Array<[MaskKey, string, Reason["thermal"], string, number]> = [
    ["skin", "skin inside person mask", "hot", "Skin-colored pixels are boosted only when they align with a face or human body mask.", 0.72],
    ["hair", "hair", "cool", "Hair is cooled relative to skin in portrait mode and person masks.", 0.62],
    ["fabric", "clothes/fabric", "cool", "Clothes inside the person mask are kept cooler than exposed skin.", 0.62],
    ["sky", "sky", "cool", `Open sky uses thermal prior ${thermalPriors.sky.toFixed(2)}.`, 0.68],
    ["cloud", "clouds", "cool", `Clouds use prior ${thermalPriors.cloud.toFixed(2)}, slightly warmer than clear sky when bright.`, 0.58],
    ["water", "water", "cool", `Water uses prior ${thermalPriors.water.toFixed(2)} for rivers, lakes, ocean-like regions.`, 0.64],
    ["snowIce", "snow/ice", "cool", `Snow and ice use very cold prior ${thermalPriors.snowIce.toFixed(2)}.`, 0.62],
    ["wetSurface", "wet surfaces", "cool", "Wet surfaces are cooled below dry ground.", 0.58],
    ["vegetation", "vegetation", "cool", `Trees, leaves, grass, and flowers stay cool-medium around prior ${thermalPriors.vegetation.toFixed(2)}.`, 0.68],
    ["forestShade", "forest shade", "cool", "Dark vegetation is cooled as forest shade.", 0.6],
    ["soil", "soil", "mixed", "Soil is moderate; dry soil is warmer than wet or shadowed soil.", 0.58],
    ["sand", "sand", "warm", "Sand is warm-hot in sunlit areas.", 0.58],
    ["rock", "rocks", "warm", "Rocks are moderate-warm depending on sunlight.", 0.54],
    ["asphalt", "asphalt", "warm", `Asphalt uses warm sunlight-sensitive prior ${thermalPriors.asphalt.toFixed(2)}.`, 0.58],
    ["concrete", "concrete", "mixed", `Concrete uses moderate prior ${thermalPriors.concrete.toFixed(2)}.`, 0.58],
    ["brick", "bricks", "mixed", "Brick surfaces are treated as moderate building material.", 0.54],
    ["roof", "roofs", "warm", "Roofs are allowed to become sun-heated warmer than walls.", 0.56],
    ["wall", "walls", "mixed", "Wall-like neutral areas stay near baseline because indoor illumination does not equal heat.", 0.58],
    ["window", "windows", "cool", "Window-like regions are treated as cooler reflective surfaces.", 0.6],
    ["glassMetal", "glass/metal", "mixed", "Reflective glass and metal are treated cautiously; bright highlights are not automatically hot.", 0.6],
    ["lamp", "ceiling lights/lamps", "mixed", "Very bright indoor lights are neutralized so they do not become false hot objects.", 0.66],
    ["shadow", "shadows", "cool", "Dark low-saturation regions are cooled as likely shadow.", 0.62]
  ];

  for (const [key, label, thermal, reason, confidence] of materialReasons) {
    if (stats[key] / total > 0.01) reasons.push({ label, confidence, thermal, reason });
  }

  return reasons.slice(0, 12);
}

async function generateThermalPrediction(
  imageUrl: string,
  settings: Settings,
  models: LoadedModels | null
): Promise<ThermalResult> {
  const image = await imageFromUrl(imageUrl);
  const maxSide = 980;
  const scale = Math.min(1, maxSide / Math.max(image.naturalWidth, image.naturalHeight));
  const width = Math.max(1, Math.round(image.naturalWidth * scale));
  const height = Math.max(1, Math.round(image.naturalHeight * scale));
  const pixelCount = width * height;

  const source = document.createElement("canvas");
  source.width = width;
  source.height = height;
  const src = source.getContext("2d", { willReadFrequently: true });
  if (!src) throw new Error("Canvas is unavailable.");
  src.drawImage(image, 0, 0, width, height);

  const rawDetections = models?.detector ? await models.detector.detect(source) : [];
  const detections = normalizeDetections(rawDetections, width, height);
  const faces = models?.faceDetector ? await models.faceDetector.estimateFaces(source) : [];
  const personDetections = detections.filter(isHumanLikeDetection);
  const animalDetections = detections.filter(isStrongAnimalDetection);
  const vehicleDetections = detections.filter(isStrongVehicleDetection);
  let personSegmentation: Uint8Array | Int32Array | null = null;
  let usedBodySegmentation = false;

  if (models?.bodyPix && personDetections.length > 0) {
    try {
      const segmentation = await models.bodyPix.segmentPerson(source, {
        internalResolution: "medium",
        segmentationThreshold: 0.74
      });
      personSegmentation = segmentation.data;
      usedBodySegmentation = true;
    } catch {
      personSegmentation = null;
    }
  }

  const data = src.getImageData(0, 0, width, height);
  const quality = imageQualityScore(data);
  const illumination = buildIlluminationMap(data, width, height);
  const masks = createMasks(pixelCount);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const p = y * width + x;
      const i = p * 4;
      const features = pixelFeatures(data.data[i], data.data[i + 1], data.data[i + 2], x, y, width, height);
      if (features.sky) masks.sky[p] = 1;
      if (features.cloud && !masks.sky[p]) masks.cloud[p] = 1;
      if (features.fog && !masks.sky[p]) masks.fog[p] = 1;
      if (features.snowIce) masks.snowIce[p] = 1;
      if (features.water && !masks.sky[p]) masks.water[p] = 1;
      if (features.wetSurface && !masks.water[p]) masks.wetSurface[p] = 1;
      if (features.vegetation) masks.vegetation[p] = 1;
      if (features.leaves) masks.leaves[p] = 1;
      if (features.grass) masks.grass[p] = 1;
      if (features.tree) masks.tree[p] = 1;
      if (features.flower) masks.flower[p] = 1;
      if (features.forestShade) masks.forestShade[p] = 1;
      if (features.soil) masks.soil[p] = 1;
      if (features.drySoil) masks.drySoil[p] = 1;
      if (features.sand) masks.sand[p] = 1;
      if (features.rock) masks.rock[p] = 1;
      if (features.asphalt) masks.asphalt[p] = 1;
      if (features.concrete) masks.concrete[p] = 1;
      if (features.brick) masks.brick[p] = 1;
      if (features.lamp) masks.lamp[p] = 1;
      if (features.window) masks.window[p] = 1;
      if (features.shadow) masks.shadow[p] = 1;
      if (features.glassMetal) masks.glassMetal[p] = 1;
      if (features.ground) masks.ground[p] = 1;
      if (features.roof) masks.roof[p] = 1;
      if (features.wall) masks.wall[p] = 1;
      if (features.sunlitSurface) masks.sunlitSurface[p] = 1;
    }
  }

  for (const detection of animalDetections) applyEllipseMask(masks.animal, detection, width, height, data.data, 0.18);
  for (const detection of vehicleDetections) applyVehicleMask(masks.vehicle, detection, width, height, data.data);

  if (personSegmentation && personDetections.length > 0) {
    for (const detection of personDetections) {
      const [nx, ny, nw, nh] = detection.bbox;
      const x0 = Math.max(0, Math.floor(nx * width));
      const y0 = Math.max(0, Math.floor(ny * height));
      const x1 = Math.min(width - 1, Math.ceil((nx + nw) * width));
      const y1 = Math.min(height - 1, Math.ceil((ny + nh) * height));
      for (let y = y0; y <= y1; y += 1) {
        for (let x = x0; x <= x1; x += 1) {
          const p = y * width + x;
          if (personSegmentation[p] !== 1) continue;
          if (masks.sky[p] || masks.water[p] || masks.vegetation[p]) continue;
          masks.person[p] = 1;
        }
      }
    }
  } else {
    for (const detection of personDetections) applyEllipseMask(masks.person, detection, width, height, data.data, 0.24);
  }

  for (const face of faces) {
    applyEllipseMask(masks.face, { bbox: face.bbox, class: "face", score: face.score }, width, height, data.data, 0.22);
  }

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const p = y * width + x;
      if (!masks.person[p] && !masks.face[p]) continue;
      const i = p * 4;
      const features = pixelFeatures(data.data[i], data.data[i + 1], data.data[i + 2], x, y, width, height);
      if (features.skin && (masks.person[p] || masks.face[p])) masks.skin[p] = 1;
      else if (features.hair && masks.person[p]) masks.hair[p] = 1;
      else if (masks.person[p] && !masks.face[p]) masks.fabric[p] = 1;
    }
  }

  const smoothKeys: MaskKey[] = [
    "face",
    "skin",
    "person",
    "hair",
    "fabric",
    "animal",
    "vehicle",
    "vehicleHotspot",
    "sky",
    "cloud",
    "fog",
    "water",
    "wetSurface",
    "snowIce",
    "vegetation",
    "leaves",
    "grass",
    "tree",
    "flower",
    "forestShade",
    "soil",
    "drySoil",
    "sand",
    "rock",
    "asphalt",
    "concrete",
    "brick",
    "ground",
    "wall",
    "roof",
    "window",
    "glassMetal",
    "lamp",
    "shadow"
  ];
  for (const key of smoothKeys) smoothMaskEdgeAware(masks[key], data.data, width, height, key === "person" ? 2 : 1);

  const out = src.createImageData(width, height);
  const heat = new Float32Array(pixelCount);
  const humanBoost = settings.humanHeatBoost / 100;
  const sunlightBoost = settings.sunlightBoost / 100;
  const contrast = 0.75 + settings.thermalContrast / 110;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const p = y * width + x;
      const i = p * 4;
      const features = pixelFeatures(data.data[i], data.data[i + 1], data.data[i + 2], x, y, width, height);
      const normalizedLight = clamp(0.5 + (illumination.brightness[p] - illumination.local[p]) * 0.72);
      const dominant = dominantMask(masks, p);
      let value = dominant ? thermalPriors[dominant] : 0.4;
      const lightAdjustment = (normalizedLight - 0.5) * 0.08;

      if (dominant && !["sky", "water", "snowIce", "window", "glassMetal", "lamp", "face", "skin", "person", "animal"].includes(dominant)) {
        value += lightAdjustment;
      }

      if (masks.cloud[p] && masks.sunlitSurface[p]) value = Math.max(value, thermalPriors.cloud + 0.06);
      if (masks.sand[p] && masks.sunlitSurface[p]) value = Math.max(value, thermalPriors.sand + sunlightBoost * 0.12);
      if (masks.rock[p] && masks.sunlitSurface[p]) value = Math.max(value, thermalPriors.rock + sunlightBoost * 0.08);
      if (masks.asphalt[p] && masks.sunlitSurface[p]) value = Math.max(value, thermalPriors.asphalt + sunlightBoost * 0.11);
      if (masks.concrete[p] && masks.sunlitSurface[p]) value = Math.max(value, thermalPriors.concrete + sunlightBoost * 0.08);
      if (masks.roof[p] && masks.sunlitSurface[p]) value = Math.max(value, thermalPriors.roof + sunlightBoost * 0.1);
      if (masks.wetSurface[p]) value = Math.min(value, thermalPriors.wetSurface);
      if (masks.shadow[p]) value = Math.min(value, thermalPriors.shadow + features.brightness * 0.08);
      if (masks.lamp[p]) value = Math.min(value, thermalPriors.lamp);
      if (masks.glassMetal[p] || masks.window[p]) value = Math.min(value, thermalPriors[dominant === "window" ? "window" : "glassMetal"] + normalizedLight * 0.04);
      if (masks.animal[p]) value = Math.max(value, thermalPriors.animal + humanBoost * 0.08);
      if (masks.person[p]) value = Math.max(value, settings.humanMode ? 0.56 : thermalPriors.person - 0.08);
      if (masks.fabric[p]) value = Math.min(Math.max(value, thermalPriors.fabric), 0.54);
      if (masks.hair[p]) value = Math.min(Math.max(value, thermalPriors.hair), settings.humanMode ? 0.45 : 0.48);
      if (masks.skin[p]) value = Math.max(value, thermalPriors.skin + humanBoost * 0.06);
      if (masks.face[p]) {
        const faceHeat = faces.reduce((max, face) => Math.max(max, faceZoneHeat(face, x, y, width, height)), 0);
        value = Math.max(value, Math.max(faceHeat, thermalPriors.face - 0.05) + humanBoost * 0.06);
      }
      if (masks.vehicle[p]) {
        const detection = vehicleDetections.find((item) => {
          const [nx, ny, nw, nh] = item.bbox;
          return x >= nx * width && x <= (nx + nw) * width && y >= ny * height && y <= (ny + nh) * height;
        });
        const hotspot = detection ? vehicleHotspot(x, y, detection, width, height) : 0;
        if (hotspot > 0.5) masks.vehicleHotspot[p] = 1;
        const cooling = detection ? depthCoolingFromDetection(detection) : 0;
        value = Math.max(value, thermalPriors.vehicle - cooling + hotspot * 0.18 + (masks.sunlitSurface[p] ? sunlightBoost * 0.05 : 0));
      }

      heat[p] = clamp((value - 0.5) * contrast + 0.5);
    }
  }

  const smoothHeat = new Float32Array(heat);
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const p = y * width + x;
      const i = p * 4;
      let sum = heat[p] * 3;
      let weight = 3;
      for (let oy = -1; oy <= 1; oy += 1) {
        for (let ox = -1; ox <= 1; ox += 1) {
          if (ox === 0 && oy === 0) continue;
          const np = (y + oy) * width + x + ox;
          const ni = np * 4;
          if (colorDistance(data.data, i, ni) > 0.18) continue;
          sum += heat[np];
          weight += 1;
        }
      }
      smoothHeat[p] = sum / weight;
    }
  }

  let totalHeat = 0;
  for (let p = 0; p < pixelCount; p += 1) {
    const i = p * 4;
    const [hr, hg, hb] = ironbow(smoothHeat[p]);
    const structure = 0.18 + illumination.brightness[p] * 0.1;
    const blend = clamp(0.66 + settings.thermalContrast / 620);
    out.data[i] = Math.round(hr * blend + data.data[i] * structure);
    out.data[i + 1] = Math.round(hg * blend + data.data[i + 1] * structure);
    out.data[i + 2] = Math.round(hb * blend + data.data[i + 2] * structure);
    out.data[i + 3] = data.data[i + 3];
    totalHeat += smoothHeat[p];
  }

  src.putImageData(out, 0, 0);
  const stats = countMasks(masks);
  const objectConfidence = detections.length
    ? detections.reduce((sum, item) => sum + item.score, 0) / detections.length
    : faces.length
      ? faces.reduce((sum, item) => sum + item.score, 0) / faces.length
      : 0.32;
  const materialCoverage =
    (stats.sky +
      stats.cloud +
      stats.fog +
      stats.water +
      stats.wetSurface +
      stats.snowIce +
      stats.vegetation +
      stats.soil +
      stats.sand +
      stats.rock +
      stats.asphalt +
      stats.concrete +
      stats.brick +
      stats.ground +
      stats.wall +
      stats.roof +
      stats.window +
      stats.glassMetal +
      stats.shadow) /
    pixelCount;
  const confidence = clamp(
    objectConfidence * 0.42 + quality * 0.28 + Math.min(materialCoverage, 0.55) * 0.22 + (usedBodySegmentation ? 0.08 : 0) + (faces.length ? 0.08 : 0)
  );
  const lowConfidence = confidence < 0.52 || (detections.length === 0 && quality < 0.58);

  return {
    url: source.toDataURL("image/png"),
    debugUrl: buildDebugMaskUrl(masks, width, height),
    stats,
    avgHeat: totalHeat / pixelCount,
    pixelCount,
    detections,
    faces,
    reasons: buildReasons(detections, faces, stats, pixelCount, lowConfidence, settings.humanMode),
    confidence,
    imageQuality: quality,
    usedBodySegmentation,
    lowConfidence
  };
}

function percent(count: number, total: number) {
  if (!total) return "0%";
  return `${Math.round((count / total) * 100)}%`;
}

function score(value: number) {
  return `${Math.round(value * 100)}%`;
}

function ThermometerIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" className="h-5 w-5">
      <path
        d="M14 14.76V5a4 4 0 0 0-8 0v9.76a6 6 0 1 0 8 0ZM10 4a2 2 0 0 1 2 2v10.7l.46.3A4 4 0 1 1 7.54 17l.46-.3V6a2 2 0 0 1 2-2Z"
        fill="currentColor"
      />
    </svg>
  );
}

export default function Home() {
  const [imageUrl, setImageUrl] = useState<string>("");
  const [thermal, setThermal] = useState<ThermalResult | null>(null);
  const [settings, setSettings] = useState<Settings>(defaultSettings);
  const [cameraOn, setCameraOn] = useState(false);
  const [showDebugMasks, setShowDebugMasks] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(true);
  const [models, setModels] = useState<LoadedModels | null>(null);
  const [message, setMessage] = useState("Loading client-side AI models...");
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const totalPixels = thermal?.pixelCount ?? 0;

  useEffect(() => {
    let mounted = true;
    loadModels()
      .then((loaded) => {
        if (!mounted) return;
        setModels(loaded);
        setMessage(
          loaded.faceDetector
            ? "COCO-SSD, BodyPix, and face detection are ready."
            : "COCO-SSD and BodyPix are ready. Face detection fallback is off."
        );
      })
      .catch(() => {
        if (!mounted) return;
        setModels({ detector: null, bodyPix: null, faceDetector: null });
        setMessage("Model loading failed, so the app will use material masks only.");
      })
      .finally(() => {
        if (mounted) setIsLoadingModels(false);
      });

    return () => {
      mounted = false;
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    setThermal(null);
    setMessage(`Loaded ${file.name}.`);
  }

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraOn(true);
      setMessage("Camera ready.");
    } catch {
      setMessage("Camera permission was blocked or no webcam is available.");
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraOn(false);
    setMessage("Camera stopped.");
  }

  function captureFrame() {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.drawImage(video, 0, 0);
    setImageUrl(canvas.toDataURL("image/png"));
    setThermal(null);
    setMessage("Frame captured from webcam.");
  }

  async function handleGenerate() {
    if (!imageUrl) {
      setMessage("Add an image before generating a thermal prediction.");
      return;
    }
    setIsGenerating(true);
    setMessage("Detecting objects, refining masks, and estimating pseudo-thermal zones...");
    try {
      const result = await generateThermalPrediction(imageUrl, settings, models);
      setThermal(result);
      setMessage(result.lowConfidence ? "Low confidence scene estimate generated." : "Scene-reasoned thermal estimate generated.");
    } catch {
      setMessage("Could not process this image in the browser.");
    } finally {
      setIsGenerating(false);
    }
  }

  function updateSetting<K extends keyof Settings>(key: K, value: Settings[K]) {
    setSettings((current) => ({ ...current, [key]: value }));
  }

  return (
    <main className="min-h-screen px-4 py-5 text-slate-100 sm:px-6 lg:px-8">
      <div className="mx-auto flex max-w-7xl flex-col gap-5">
        <header className="flex flex-col justify-between gap-4 border-b border-white/10 pb-5 md:flex-row md:items-end">
          <div>
            <p className="text-sm font-bold uppercase text-cyan-300">Client-side pseudo-thermal reasoning</p>
            <h1 className="mt-2 text-4xl font-black text-white sm:text-6xl">TherA Lite</h1>
            <p className="mt-3 max-w-3xl text-base leading-7 text-slate-300">
              A browser-only RGB scene estimator using object detections, refined masks, material cues, sun/shadow logic,
              and edge-aware heat blending. It predicts relative thermal appearance, not measured temperature.
            </p>
          </div>
          <button
            onClick={handleGenerate}
            disabled={!imageUrl || isGenerating || isLoadingModels}
            className="inline-flex h-12 items-center justify-center gap-2 rounded-md bg-cyan-300 px-5 text-sm font-black text-slate-950 shadow-glow transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
          >
            <ThermometerIcon />
            {isGenerating ? "Reasoning..." : isLoadingModels ? "Loading AI..." : "Generate thermal prediction"}
          </button>
        </header>

        <section className="grid gap-5 lg:grid-cols-[360px_1fr]">
          <aside className="flex flex-col gap-4">
            <Panel title="Input">
              <label className="mt-4 flex min-h-32 cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-cyan-300/35 bg-white/[0.04] px-4 py-6 text-center transition hover:border-cyan-200 hover:bg-cyan-300/10">
                <span className="text-sm font-bold text-white">Choose image</span>
                <span className="mt-1 text-xs text-slate-400">PNG, JPG, or webcam frame</span>
                <input className="sr-only" type="file" accept="image/*" onChange={handleUpload} />
              </label>

              <div className="mt-4 overflow-hidden rounded-md border border-white/10 bg-slate-950">
                <video ref={videoRef} className="aspect-video w-full object-cover" muted playsInline />
              </div>
              <div className="mt-3 grid grid-cols-2 gap-2">
                <button
                  onClick={cameraOn ? stopCamera : startCamera}
                  className="h-10 rounded-md border border-white/15 bg-white/[0.04] text-sm font-bold text-slate-100 transition hover:border-cyan-300 hover:text-cyan-200"
                >
                  {cameraOn ? "Stop camera" : "Start camera"}
                </button>
                <button
                  onClick={captureFrame}
                  disabled={!cameraOn}
                  className="h-10 rounded-md bg-violet-400 text-sm font-black text-slate-950 transition hover:bg-violet-300 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
                >
                  Capture
                </button>
              </div>
              <p className="mt-3 min-h-5 text-sm text-slate-400">{message}</p>
            </Panel>

            <Panel title="Thermal controls">
              <Slider
                label="Human heat boost"
                value={settings.humanHeatBoost}
                onChange={(value) => updateSetting("humanHeatBoost", value)}
              />
              <Slider
                label="Sunlight boost"
                value={settings.sunlightBoost}
                onChange={(value) => updateSetting("sunlightBoost", value)}
              />
              <Slider
                label="Thermal contrast"
                value={settings.thermalContrast}
                onChange={(value) => updateSetting("thermalContrast", value)}
              />
              <label className="mt-4 flex items-center justify-between rounded-md border border-white/10 bg-white/[0.04] px-3 py-2 text-sm font-bold text-slate-200">
                Human mode
                <input
                  type="checkbox"
                  checked={settings.humanMode}
                  onChange={(event) => updateSetting("humanMode", event.target.checked)}
                  className="h-4 w-4 accent-rose-300"
                />
              </label>
              <label className="mt-4 flex items-center justify-between rounded-md border border-white/10 bg-white/[0.04] px-3 py-2 text-sm font-bold text-slate-200">
                Debug masks
                <input
                  type="checkbox"
                  checked={showDebugMasks}
                  onChange={(event) => setShowDebugMasks(event.target.checked)}
                  className="h-4 w-4 accent-cyan-300"
                />
              </label>
            </Panel>

            <Panel title="Confidence">
              <div className="mt-4 grid grid-cols-2 gap-3">
                <Metric label="Prediction" value={thermal ? score(thermal.confidence) : "--"} />
                <Metric label="Image quality" value={thermal ? score(thermal.imageQuality) : "--"} />
              </div>
              {thermal?.lowConfidence ? (
                <p className="mt-3 rounded-md border border-amber-300/30 bg-amber-300/10 p-3 text-xs font-bold leading-5 text-amber-100">
                  Low confidence scene estimate. People and animals are only shown when confident detections exist.
                </p>
              ) : (
                <p className="mt-3 text-xs leading-5 text-slate-400">
                  Confidence combines detector scores, image quality, mask coverage, and whether BodyPix helped refine people.
                </p>
              )}
            </Panel>
          </aside>

          <section className="flex flex-col gap-5">
            <div className="grid gap-5 xl:grid-cols-2">
              <ImagePanel title="Original RGB" src={imageUrl} empty="Upload or capture a scene." />
              <ImagePanel
                title={showDebugMasks ? "Debug masks" : "Thermal estimate"}
                src={showDebugMasks ? thermal?.debugUrl ?? "" : thermal?.url ?? ""}
                empty="Generate a prediction to view the output."
              />
            </div>

            <div className="grid gap-5 lg:grid-cols-[1fr_320px]">
              <Panel title="Reasoning">
                <p className="mt-3 text-sm leading-6 text-slate-400">
                  TherA Lite creates a pseudo-thermal estimate from RGB structure. It does not measure real temperature and
                  should not be used as a calibrated thermal camera.
                </p>
                <div className="mt-4 grid gap-3">
                  {(thermal?.reasons ?? []).length > 0 ? (
                    thermal?.reasons.map((item, index) => <ReasonCard key={`${item.label}-${index}`} item={item} />)
                  ) : (
                    <div className="rounded-md border border-white/10 bg-white/[0.04] p-3 text-sm leading-6 text-slate-400">
                      Confident detections and material-mask decisions will appear here after prediction.
                    </div>
                  )}
                </div>
              </Panel>

              <Panel title="Detected cues">
                <div className="mt-4 flex flex-col gap-3">
                  <Cue label="Faces" value={thermal ? percent(thermal.stats.face, totalPixels) : "0%"} color="bg-orange-300" />
                  <Cue label="Skin" value={thermal ? percent(thermal.stats.skin, totalPixels) : "0%"} color="bg-rose-300" />
                  <Cue label="People" value={thermal ? percent(thermal.stats.person, totalPixels) : "0%"} color="bg-rose-400" />
                  <Cue label="Hair" value={thermal ? percent(thermal.stats.hair, totalPixels) : "0%"} color="bg-slate-700" />
                  <Cue label="Clothes" value={thermal ? percent(thermal.stats.fabric, totalPixels) : "0%"} color="bg-violet-400" />
                  <Cue label="Animals" value={thermal ? percent(thermal.stats.animal, totalPixels) : "0%"} color="bg-orange-300" />
                  <Cue label="Vehicles" value={thermal ? percent(thermal.stats.vehicle, totalPixels) : "0%"} color="bg-amber-300" />
                  <Cue label="Vehicle hotspots" value={thermal ? percent(thermal.stats.vehicleHotspot, totalPixels) : "0%"} color="bg-yellow-100" />
                  <Cue label="Sky" value={thermal ? percent(thermal.stats.sky, totalPixels) : "0%"} color="bg-cyan-300" />
                  <Cue label="Clouds/fog" value={thermal ? percent(thermal.stats.cloud + thermal.stats.fog, totalPixels) : "0%"} color="bg-slate-300" />
                  <Cue label="Water" value={thermal ? percent(thermal.stats.water, totalPixels) : "0%"} color="bg-blue-400" />
                  <Cue label="Wet/snow/ice" value={thermal ? percent(thermal.stats.wetSurface + thermal.stats.snowIce, totalPixels) : "0%"} color="bg-blue-200" />
                  <Cue label="Vegetation" value={thermal ? percent(thermal.stats.vegetation, totalPixels) : "0%"} color="bg-emerald-400" />
                  <Cue label="Grass/leaves/trees" value={thermal ? percent(thermal.stats.grass + thermal.stats.leaves + thermal.stats.tree, totalPixels) : "0%"} color="bg-green-400" />
                  <Cue label="Forest shade" value={thermal ? percent(thermal.stats.forestShade, totalPixels) : "0%"} color="bg-green-800" />
                  <Cue label="Soil/sand/rock" value={thermal ? percent(thermal.stats.soil + thermal.stats.drySoil + thermal.stats.sand + thermal.stats.rock, totalPixels) : "0%"} color="bg-yellow-700" />
                  <Cue label="Asphalt/concrete" value={thermal ? percent(thermal.stats.asphalt + thermal.stats.concrete, totalPixels) : "0%"} color="bg-neutral-400" />
                  <Cue label="Bricks/roofs" value={thermal ? percent(thermal.stats.brick + thermal.stats.roof, totalPixels) : "0%"} color="bg-orange-700" />
                  <Cue label="Ground" value={thermal ? percent(thermal.stats.ground, totalPixels) : "0%"} color="bg-stone-400" />
                  <Cue label="Walls" value={thermal ? percent(thermal.stats.wall, totalPixels) : "0%"} color="bg-zinc-400" />
                  <Cue label="Windows" value={thermal ? percent(thermal.stats.window, totalPixels) : "0%"} color="bg-sky-300" />
                  <Cue label="Glass/metal" value={thermal ? percent(thermal.stats.glassMetal, totalPixels) : "0%"} color="bg-sky-200" />
                  <Cue label="Lights/lamps" value={thermal ? percent(thermal.stats.lamp, totalPixels) : "0%"} color="bg-yellow-100" />
                  <Cue
                    label="Sunlit surfaces"
                    value={thermal ? percent(thermal.stats.sunlitSurface, totalPixels) : "0%"}
                    color="bg-yellow-300"
                  />
                  <Cue label="Shadows" value={thermal ? percent(thermal.stats.shadow, totalPixels) : "0%"} color="bg-slate-500" />
                </div>
                <div className="mt-4 rounded-md border border-white/10 bg-white/[0.04] p-3 text-xs leading-5 text-slate-400">
                  Body segmentation: {thermal ? (thermal.usedBodySegmentation ? "active" : "not active") : "--"}
                </div>
              </Panel>
            </div>
          </section>
        </section>
      </div>
    </main>
  );
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="rounded-lg border border-white/10 bg-slate-900/72 p-4 shadow-deep backdrop-blur-xl">
      <h2 className="text-lg font-black text-white">{title}</h2>
      {children}
    </div>
  );
}

function Slider({
  label,
  value,
  onChange
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="mt-4 block">
      <span className="flex items-center justify-between text-sm font-bold text-slate-200">
        {label}
        <span className="text-cyan-200">{value}</span>
      </span>
      <input
        type="range"
        min="0"
        max="100"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-2 w-full accent-cyan-300"
      />
    </label>
  );
}

function ImagePanel({ title, src, empty }: { title: string; src: string; empty: string }) {
  return (
    <div className="rounded-lg border border-white/10 bg-slate-900/72 p-4 shadow-deep backdrop-blur-xl">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-lg font-black text-white">{title}</h2>
      </div>
      <div className="checkerboard flex aspect-[4/3] items-center justify-center overflow-hidden rounded-md border border-white/10 bg-slate-950">
        {src ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={src} alt={title} className="h-full w-full object-contain" />
        ) : (
          <p className="px-6 text-center text-sm text-slate-500">{empty}</p>
        )}
      </div>
    </div>
  );
}

function ReasonCard({ item }: { item: Reason }) {
  const tone =
    item.thermal === "hot"
      ? "border-rose-400/30 bg-rose-400/10 text-rose-200"
      : item.thermal === "warm"
        ? "border-amber-300/30 bg-amber-300/10 text-amber-100"
        : item.thermal === "cool"
          ? "border-cyan-300/30 bg-cyan-300/10 text-cyan-100"
          : "border-violet-300/30 bg-violet-300/10 text-violet-100";

  return (
    <div className={`rounded-md border p-3 ${tone}`}>
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-black capitalize">{item.label}</p>
        <p className="text-xs font-bold">{score(item.confidence)}</p>
      </div>
      <p className="mt-2 text-sm leading-6 text-slate-300">{item.reason}</p>
    </div>
  );
}

function Cue({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-md border border-white/10 bg-white/[0.04] px-3 py-2">
      <span className="flex min-w-0 items-center gap-2 text-sm text-slate-300">
        <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${color}`} />
        <span className="truncate">{label}</span>
      </span>
      <span className="text-sm font-black text-white">{value}</span>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-white/10 bg-white/[0.04] p-3">
      <p className="text-xs font-bold text-slate-400">{label}</p>
      <p className="mt-1 text-2xl font-black text-white">{value}</p>
    </div>
  );
}
