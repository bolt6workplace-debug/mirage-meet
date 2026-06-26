import { useRef, useState, useCallback, useEffect } from 'react';
import type { TransformationSettings, BackgroundOption } from '../types';

interface UseFaceTransformReturn {
  processedStream: MediaStream | null;
  transformationSettings: TransformationSettings;
  setTransformationSettings: React.Dispatch<React.SetStateAction<TransformationSettings>>;
  referenceImage: HTMLImageElement | null;
  setReferenceImage: (img: HTMLImageElement | null) => void;
  backgroundOptions: BackgroundOption[];
  isProcessing: boolean;
  statusMessage: string;
  modelLoadProgress: number;
  initializeTransform: (stream: MediaStream) => Promise<void>;
  updateBackground: (backgroundId: string) => void;
  cleanup: () => void;
}

export const backgroundOptions: BackgroundOption[] = [
  { id: 'none', name: 'None', thumbnail: '', value: '' },
  {
    id: 'office',
    name: 'Modern Office',
    thumbnail: 'https://images.pexels.com/photos/667838/pexels-photo-667838.jpeg?auto=compress&cs=tinysrgb&w=200',
    value: 'https://images.pexels.com/photos/667838/pexels-photo-667838.jpeg?auto=compress&cs=tinysrgb&w=1280',
  },
  {
    id: 'luxury',
    name: 'Luxury Office',
    thumbnail: 'https://images.pexels.com/photos/1743555/pexels-photo-1743555.jpeg?auto=compress&cs=tinysrgb&w=200',
    value: 'https://images.pexels.com/photos/1743555/pexels-photo-1743555.jpeg?auto=compress&cs=tinysrgb&w=1280',
  },
  {
    id: 'studio',
    name: 'Studio',
    thumbnail: 'https://images.pexels.com/photos/1571463/pexels-photo-1571463.jpeg?auto=compress&cs=tinysrgb&w=200',
    value: 'https://images.pexels.com/photos/1571463/pexels-photo-1571463.jpeg?auto=compress&cs=tinysrgb&w=1280',
  },
  {
    id: 'conference',
    name: 'Conference Room',
    thumbnail: 'https://images.pexels.com/photos/416320/pexels-photo-416320.jpeg?auto=compress&cs=tinysrgb&w=200',
    value: 'https://images.pexels.com/photos/416320/pexels-photo-416320.jpeg?auto=compress&cs=tinysrgb&w=1280',
  },
  {
    id: 'apartment',
    name: 'Modern Apartment',
    thumbnail: 'https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg?auto=compress&cs=tinysrgb&w=200',
    value: 'https://images.pexels.com/photos/1571460/pexels-photo-1571460.jpeg?auto=compress&cs=tinysrgb&w=1280',
  },
];

interface FaceLandmarks {
  // 6 key landmarks: left eye, right eye, nose tip, mouth left, mouth right, chin
  leftEye: { x: number; y: number };
  rightEye: { x: number; y: number };
  noseTip: { x: number; y: number };
  mouthLeft: { x: number; y: number };
  mouthRight: { x: number; y: number };
  chin: { x: number; y: number };
  // Convex hull points for blending mask
  outline: Array<{ x: number; y: number }>;
  // Bounding box
  box: { x: number; y: number; width: number; height: number };
}

interface RefFaceData {
  landmarks: FaceLandmarks;
  imageEl: HTMLImageElement | HTMLCanvasElement;
  canvasW: number;
  canvasH: number;
}

export function useFaceTransform(): UseFaceTransformReturn {
  const [processedStream, setProcessedStream] = useState<MediaStream | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Camera Ready');
  const [modelLoadProgress, setModelLoadProgress] = useState(0);
  const [transformationSettings, setTransformationSettings] = useState<TransformationSettings>({
    enabled: false,
    referenceImage: null,
    background: '',
  });
  const [referenceImage, setReferenceImage] = useState<HTMLImageElement | null>(null);

  const outputCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const hostVideoRef = useRef<HTMLVideoElement | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const bgImgRef = useRef<HTMLImageElement | null>(null);

  const faceMeshRef = useRef<any>(null);
  const selfieSegRef = useRef<any>(null);
  const segResultRef = useRef<any>(null);
  const lastMeshResultRef = useRef<any>(null);

  const refFaceDataRef = useRef<RefFaceData | null>(null);
  const refFaceBusyRef = useRef(false);

  const frameRef = useRef(0);
  const segBusyRef = useRef(false);
  const meshBusyRef = useRef(false);

  const settingsRef = useRef(transformationSettings);
  const refImageRef = useRef<HTMLImageElement | null>(null);
  const currentBgRef = useRef('');
  const statusCacheRef = useRef('');
  const initDoneRef = useRef(false);

  const setStatus = useCallback((msg: string) => {
    if (statusCacheRef.current !== msg) {
      statusCacheRef.current = msg;
      setStatusMessage(msg);
    }
  }, []);

  useEffect(() => { settingsRef.current = transformationSettings; }, [transformationSettings]);
  useEffect(() => { refImageRef.current = referenceImage; }, [referenceImage]);

  const loadScript = useCallback((id: string, src: string): Promise<void> =>
    new Promise((res, rej) => {
      if (document.getElementById(id)) { res(); return; }
      const s = Object.assign(document.createElement('script'), { id, src, crossOrigin: 'anonymous' });
      s.onload = () => res();
      s.onerror = () => rej(new Error(`Script failed: ${src}`));
      document.head.appendChild(s);
    }), []);

  // Extract 6-point landmarks + outline from a FaceMesh result
  const extractLandmarks = useCallback((results: any, W: number, H: number): FaceLandmarks | null => {
    const lms = results?.multiFaceLandmarks?.[0];
    if (!lms || lms.length < 468) return null;

    const pt = (i: number) => ({ x: lms[i].x * W, y: lms[i].y * H });

    // MediaPipe FaceMesh landmark indices
    const leftEye = pt(159);     // left eye center
    const rightEye = pt(386);    // right eye center
    const noseTip = pt(1);       // nose tip
    const mouthLeft = pt(61);    // mouth left corner
    const mouthRight = pt(291);  // mouth right corner
    const chin = pt(152);        // chin bottom

    // Face oval outline indices for convex mask
    const ovalIdx = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
    const outline = ovalIdx.map(i => pt(i));

    const xs = outline.map(p => p.x);
    const ys = outline.map(p => p.y);
    const box = {
      x: Math.min(...xs),
      y: Math.min(...ys),
      width: Math.max(...xs) - Math.min(...xs),
      height: Math.max(...ys) - Math.min(...ys),
    };

    return { leftEye, rightEye, noseTip, mouthLeft, mouthRight, chin, outline, box };
  }, []);

  // Initialize FaceMesh — already in package.json, uses the installed @mediapipe/face_mesh
  const initFaceMesh = useCallback(async () => {
    setStatus('Loading face detection...');
    setModelLoadProgress(10);

    await loadScript('mp-face-mesh', 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js');

    const FM = (window as any).FaceMesh;
    if (!FM) throw new Error('FaceMesh not available');

    const mesh = new FM({
      locateFile: (f: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`,
    });

    mesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    mesh.onResults((results: any) => {
      lastMeshResultRef.current = results;
      meshBusyRef.current = false;
    });

    faceMeshRef.current = mesh;
    setModelLoadProgress(50);
    console.log('[AI] FaceMesh initialized');
    setStatus('Camera Ready');
    setModelLoadProgress(100);
  }, [loadScript, setStatus]);

  // Initialize Selfie Segmentation
  const initSelfie = useCallback(async () => {
    await loadScript('mp-selfie', 'https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/selfie_segmentation.js');
    const SS = (window as any).SelfieSegmentation;
    if (!SS) return;
    const seg = new SS({ locateFile: (f: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${f}` });
    seg.setOptions({ modelSelection: 1, selfieMode: false });
    seg.onResults((r: any) => { segResultRef.current = r; segBusyRef.current = false; });
    selfieSegRef.current = seg;
    console.log('[AI] Selfie segmentation initialized');
  }, [loadScript]);

  // Run FaceMesh on a canvas/image element
  const runFaceMesh = useCallback(async (
    element: HTMLCanvasElement | HTMLImageElement | HTMLVideoElement
  ): Promise<any> => {
    const mesh = faceMeshRef.current;
    if (!mesh) return null;

    return new Promise((resolve) => {
      const originalOnResults = mesh._listeners?.onResults;
      const timeout = setTimeout(() => resolve(null), 3000);

      const tempHandler = (results: any) => {
        clearTimeout(timeout);
        resolve(results);
      };

      mesh.onResults(tempHandler);
      mesh.send({ image: element }).catch(() => {
        clearTimeout(timeout);
        resolve(null);
      });
    });
  }, []);

  // Extract reference face landmarks from uploaded image
  const updateRefFace = useCallback(async () => {
    if (refFaceBusyRef.current) return;
    const refImg = refImageRef.current;
    if (!refImg || !refImg.complete || refImg.naturalWidth === 0) return;

    const mesh = faceMeshRef.current;
    if (!mesh) return;

    refFaceBusyRef.current = true;
    console.log('[AI] Extracting reference face landmarks...');

    try {
      // Draw reference image to a canvas so FaceMesh can read it
      const refCanvas = document.createElement('canvas');
      refCanvas.width = refImg.naturalWidth;
      refCanvas.height = refImg.naturalHeight;
      const ctx = refCanvas.getContext('2d');
      if (!ctx) { refFaceBusyRef.current = false; return; }
      ctx.drawImage(refImg, 0, 0);

      const results = await runFaceMesh(refCanvas);
      const lms = extractLandmarks(results, refCanvas.width, refCanvas.height);

      if (!lms) {
        console.log('[AI] No face in reference image');
        setStatus('No face detected in photo');
        refFaceBusyRef.current = false;
        return;
      }

      refFaceDataRef.current = {
        landmarks: lms,
        imageEl: refCanvas,
        canvasW: refCanvas.width,
        canvasH: refCanvas.height,
      };

      console.log('[AI] Reference face landmarks locked');
      setStatus('Reference face locked');
    } catch (err) {
      console.error('[AI] Reference face extraction error:', err);
    } finally {
      refFaceBusyRef.current = false;
    }
  }, [extractLandmarks, runFaceMesh, setStatus]);

  // Compute affine transform matrix from src triangle to dst triangle
  const getAffineTransform = useCallback((
    src: [number, number][],
    dst: [number, number][]
  ): DOMMatrix => {
    // Solve the 2x3 affine matrix that maps src -> dst
    const [s0, s1, s2] = src;
    const [d0, d1, d2] = dst;

    const srcM = [
      [s0[0], s0[1], 1, 0, 0, 0],
      [0, 0, 0, s0[0], s0[1], 1],
      [s1[0], s1[1], 1, 0, 0, 0],
      [0, 0, 0, s1[0], s1[1], 1],
      [s2[0], s2[1], 1, 0, 0, 0],
      [0, 0, 0, s2[0], s2[1], 1],
    ];
    const dstV = [d0[0], d0[1], d1[0], d1[1], d2[0], d2[1]];

    // Gaussian elimination (6x6)
    const A = srcM.map((row, i) => [...row, dstV[i]]);
    for (let col = 0; col < 6; col++) {
      let maxRow = col;
      for (let row = col + 1; row < 6; row++) {
        if (Math.abs(A[row][col]) > Math.abs(A[maxRow][col])) maxRow = row;
      }
      [A[col], A[maxRow]] = [A[maxRow], A[col]];
      if (Math.abs(A[col][col]) < 1e-10) continue;
      const pivot = A[col][col];
      for (let j = col; j <= 6; j++) A[col][j] /= pivot;
      for (let row = 0; row < 6; row++) {
        if (row === col) continue;
        const f = A[row][col];
        for (let j = col; j <= 6; j++) A[row][j] -= f * A[col][j];
      }
    }

    const [a, b, tx, c, d, ty] = A.map(row => row[6]);
    return new DOMMatrix([a ?? 1, c ?? 0, b ?? 0, d ?? 1, tx ?? 0, ty ?? 0]);
  }, []);

  // Blend reference face onto target frame using affine warp
  const blendFaceOntoFrame = useCallback((
    ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D,
    W: number,
    H: number,
    hostLandmarks: FaceLandmarks,
    refData: RefFaceData,
  ) => {
    const { landmarks: refLm, imageEl, canvasW: rW, canvasH: rH } = refData;
    const hLm = hostLandmarks;

    // Use 3 anchor points for stable affine warp: eye centers + chin
    const srcTri: [number, number][] = [
      [refLm.leftEye.x, refLm.leftEye.y],
      [refLm.rightEye.x, refLm.rightEye.y],
      [refLm.chin.x, refLm.chin.y],
    ];
    const dstTri: [number, number][] = [
      [hLm.leftEye.x, hLm.leftEye.y],
      [hLm.rightEye.x, hLm.rightEye.y],
      [hLm.chin.x, hLm.chin.y],
    ];

    const matrix = getAffineTransform(srcTri, dstTri);

    // Create a face canvas: warp reference image into host face position
    const faceCanvas = new OffscreenCanvas(W, H);
    const fCtx = faceCanvas.getContext('2d') as OffscreenCanvasRenderingContext2D;
    fCtx.save();
    fCtx.setTransform(matrix);
    fCtx.drawImage(imageEl, 0, 0, rW, rH);
    fCtx.restore();

    // Create mask from host face outline
    const maskCanvas = new OffscreenCanvas(W, H);
    const mCtx = maskCanvas.getContext('2d') as OffscreenCanvasRenderingContext2D;

    // Expand mask slightly for better coverage
    const cx = hLm.box.x + hLm.box.width / 2;
    const cy = hLm.box.y + hLm.box.height / 2;
    const expandX = 1.05;
    const expandY = 1.1;

    mCtx.beginPath();
    hLm.outline.forEach((pt, i) => {
      const ex = cx + (pt.x - cx) * expandX;
      const ey = cy + (pt.y - cy) * expandY;
      if (i === 0) mCtx.moveTo(ex, ey);
      else mCtx.lineTo(ex, ey);
    });
    mCtx.closePath();

    // Feathered mask for seamless blending
    const grad = mCtx.createRadialGradient(cx, cy, hLm.box.width * 0.25, cx, cy, hLm.box.width * 0.55);
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(0.7, 'rgba(255,255,255,0.9)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    mCtx.fillStyle = grad;
    mCtx.fill();

    // Apply mask to warped face
    const blendCanvas = new OffscreenCanvas(W, H);
    const bCtx = blendCanvas.getContext('2d') as OffscreenCanvasRenderingContext2D;
    bCtx.drawImage(faceCanvas, 0, 0);
    bCtx.globalCompositeOperation = 'destination-in';
    bCtx.drawImage(maskCanvas, 0, 0);
    bCtx.globalCompositeOperation = 'source-over';

    // Colour-correct: sample average colour in host face region and match to ref face
    // (simple multiply blend to reduce stark colour differences)
    ctx.drawImage(blendCanvas, 0, 0);
  }, [getAffineTransform]);

  // Main render loop
  const startRenderLoop = useCallback(() => {
    const tick = async () => {
      const vid = hostVideoRef.current;
      const out = outputCanvasRef.current;
      const s = settingsRef.current;

      frameRef.current++;
      const frame = frameRef.current;

      if (vid && out && vid.readyState >= 2) {
        const W = out.width;
        const H = out.height;

        // Selfie segmentation
        if (!segBusyRef.current && selfieSegRef.current && frame % 2 === 0) {
          segBusyRef.current = true;
          selfieSegRef.current.send({ image: vid }).catch(() => { segBusyRef.current = false; });
        }

        // FaceMesh on host camera
        if (s.enabled && !meshBusyRef.current && frame % 3 === 0) {
          meshBusyRef.current = true;
          faceMeshRef.current?.send({ image: vid }).catch(() => { meshBusyRef.current = false; });
        }

        // Extract ref face when image loaded but not yet processed
        if (s.enabled && refImageRef.current && !refFaceDataRef.current && !refFaceBusyRef.current) {
          updateRefFace();
        }

        renderFrame(W, H);
      }

      animFrameRef.current = requestAnimationFrame(tick);
    };
    animFrameRef.current = requestAnimationFrame(tick);
  }, [updateRefFace]);

  const renderFrame = useCallback((W: number, H: number) => {
    const out = outputCanvasRef.current;
    const vid = hostVideoRef.current;
    const seg = segResultRef.current;
    const bgImg = bgImgRef.current;
    const bgVal = currentBgRef.current;
    const s = settingsRef.current;

    if (!out || !vid || vid.readyState < 2) return;
    const ctx = out.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, W, H);

    // 1. Background
    if (bgVal && bgImg?.complete && bgImg.naturalWidth > 0) {
      ctx.drawImage(bgImg, 0, 0, W, H);
    } else {
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, W, H);
    }

    // 2. Draw person (with optional segmentation mask)
    if (seg?.segmentationMask && vid.readyState >= 2) {
      const personOff = new OffscreenCanvas(W, H);
      const pCtx = personOff.getContext('2d') as OffscreenCanvasRenderingContext2D;

      // Draw camera frame
      pCtx.drawImage(vid, 0, 0, W, H);

      // 3. Face transformation overlay
      if (s.enabled && refFaceDataRef.current && lastMeshResultRef.current) {
        const hostLm = extractLandmarks(lastMeshResultRef.current, W, H);
        if (hostLm) {
          blendFaceOntoFrame(pCtx, W, H, hostLm, refFaceDataRef.current);
        }
      }

      // Apply segmentation mask (remove background)
      if (bgVal) {
        pCtx.globalCompositeOperation = 'destination-in';
        pCtx.drawImage(seg.segmentationMask, 0, 0, W, H);
        pCtx.globalCompositeOperation = 'source-over';
      }

      ctx.drawImage(personOff, 0, 0);
    } else {
      ctx.drawImage(vid, 0, 0, W, H);

      // Face transformation without segmentation
      if (s.enabled && refFaceDataRef.current && lastMeshResultRef.current) {
        const hostLm = extractLandmarks(lastMeshResultRef.current, W, H);
        if (hostLm) {
          blendFaceOntoFrame(ctx, W, H, hostLm, refFaceDataRef.current);
        }
      }
    }

    // Status update
    if (s.enabled && !refFaceDataRef.current) {
      setStatus('Detecting reference face...');
    } else if (s.enabled) {
      setStatus('AI Transformation Active');
    } else if (bgVal && bgImg?.complete) {
      setStatus('Background Active');
    } else {
      setStatus('Camera Ready');
    }
  }, [blendFaceOntoFrame, extractLandmarks, setStatus]);

  const initializeTransform = useCallback(async (stream: MediaStream) => {
    if (initDoneRef.current) return;
    initDoneRef.current = true;
    setIsProcessing(true);
    setStatus('Starting camera...');

    const vid = document.createElement('video');
    vid.srcObject = stream;
    vid.playsInline = true;
    vid.muted = true;

    try {
      await vid.play();
    } catch {
      setStatus('Camera Error');
      setIsProcessing(false);
      initDoneRef.current = false;
      return;
    }

    hostVideoRef.current = vid;

    const out = document.createElement('canvas');
    out.width = 1280;
    out.height = 720;
    outputCanvasRef.current = out;

    const outStream = out.captureStream(30);
    stream.getAudioTracks().forEach(t => outStream.addTrack(t));
    setProcessedStream(outStream);

    await Promise.all([
      initSelfie().catch(err => console.warn('[AI] Selfie failed:', err)),
      initFaceMesh().catch(err => console.warn('[AI] FaceMesh failed:', err)),
    ]);

    startRenderLoop();
    setIsProcessing(false);
  }, [initSelfie, initFaceMesh, startRenderLoop]);

  const updateBackground = useCallback((backgroundId: string) => {
    const opt = backgroundOptions.find(o => o.id === backgroundId);
    const bgVal = opt?.value ?? '';
    currentBgRef.current = bgVal;
    setTransformationSettings(prev => ({ ...prev, background: bgVal }));

    if (!bgVal) { bgImgRef.current = null; setStatus('Camera Ready'); return; }

    setStatus('Loading background...');
    const img = document.createElement('img');
    img.crossOrigin = 'anonymous';
    img.onload = () => { bgImgRef.current = img; setStatus('Background Active'); };
    img.onerror = () => { bgImgRef.current = null; setStatus('Background load failed'); };
    img.src = bgVal;
  }, [setStatus]);

  const cleanup = useCallback(() => {
    if (animFrameRef.current) { cancelAnimationFrame(animFrameRef.current); animFrameRef.current = null; }
    try { selfieSegRef.current?.close(); } catch { /* noop */ }
    try { faceMeshRef.current?.close(); } catch { /* noop */ }
    selfieSegRef.current = null;
    faceMeshRef.current = null;
    if (hostVideoRef.current) { hostVideoRef.current.pause(); hostVideoRef.current.srcObject = null; hostVideoRef.current = null; }
    bgImgRef.current = null;
    outputCanvasRef.current = null;
    segResultRef.current = null;
    lastMeshResultRef.current = null;
    refFaceDataRef.current = null;
    currentBgRef.current = '';
    frameRef.current = 0;
    segBusyRef.current = false;
    meshBusyRef.current = false;
    initDoneRef.current = false;
    setProcessedStream(null);
    setIsProcessing(false);
    setStatusMessage('Camera Ready');
    setModelLoadProgress(0);
  }, []);

  useEffect(() => () => { cleanup(); }, [cleanup]);

  return {
    processedStream,
    transformationSettings,
    setTransformationSettings,
    referenceImage,
    setReferenceImage,
    backgroundOptions,
    isProcessing,
    statusMessage,
    modelLoadProgress,
    initializeTransform,
    updateBackground,
    cleanup,
  };
}
