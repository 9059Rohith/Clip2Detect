import { useEffect, useState } from "react";
import {
  ArrowDownRight, ArrowRight, BarChart3, Box, Check, ChevronRight,
  Code2, Download, ExternalLink, Eye, EyeOff,
  Film, Layers3, Pause, Play, ScanSearch, Sparkles, TerminalSquare,
} from "lucide-react";
import demoJson from "./data/demo.json";

type Annotation = { class: "fruit" | "bomb"; xyxy: [number, number, number, number] };
type Frame = { frame: number; objects: Annotation[] };
type AiResult = { frame: number; model: string; source: string; objects: Annotation[] };
type Metrics = {
  map50: number;
  map50_95: number;
  precision: number;
  recall: number;
  target_accuracy: number;
  meets_target: boolean;
  per_class: { class: string; ap50: number }[];
};
type Demo = {
  project: string;
  source: string;
  frames: Frame[];
  objectCount: number;
  augmentedCount: number;
  trainCount: number;
  validationCount: number;
  eval: Metrics | null;
};
const demo = demoJson as unknown as Demo;

const phases = [
  { name: "Collect", detail: "Extract frames from a source video", icon: Film, text: "The Python collector opens a local video and samples frames at the requested FPS." },
  { name: "Label", detail: "Build YOLO box annotations", icon: ScanSearch, text: "This reproducible fixture uses exact geometry from its source generator. The live AI action independently labels a selected frame with OpenAI vision." },
  { name: "Augment", detail: "Create training variations", icon: Layers3, text: "Pillow and NumPy create flips, brightness, contrast, and noise variants. Boxes are transformed with the images." },
  { name: "Train", detail: "Fit an Ultralytics YOLO model", icon: Box, text: "PyTorch and Ultralytics fit YOLOv8n on the labeled dataset. Source frames and their variants stay in the same split." },
  { name: "Evaluate", detail: "Measure held-out performance", icon: BarChart3, text: "The trained weights are scored on held-out source groups and their variants. mAP, precision, and recall come from the actual evaluation JSON." },
];
const classColors: Record<string, string> = { fruit: "#baff75", bomb: "#ffad6a" };

function metric(value: number | undefined) {
  return value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
}
function frameFile(frame: number) {
  return `frame_${String(frame).padStart(6, "0")}.jpg`;
}

function AnimatedArchitecture() {
  return (
    <div className="animated-architecture">
      <div className="arch-nodes">
        <div className="arch-node">
          <div className="arch-icon"><Film size={20} /></div>
          <span>Video Source</span>
          <div className="ripple"></div>
        </div>
        <div className="arch-connector">
          <div className="particle agent-1"></div>
          <div className="particle agent-2"></div>
        </div>
        <div className="arch-node">
          <div className="arch-icon"><ScanSearch size={20} /></div>
          <span>AI Labeling</span>
          <div className="ripple"></div>
        </div>
        <div className="arch-connector">
          <div className="particle agent-3"></div>
        </div>
        <div className="arch-node">
          <div className="arch-icon"><Layers3 size={20} /></div>
          <span>Augmentation</span>
          <div className="ripple"></div>
        </div>
        <div className="arch-connector">
          <div className="particle agent-1"></div>
          <div className="particle agent-4"></div>
        </div>
        <div className="arch-node">
          <div className="arch-icon"><Box size={20} /></div>
          <span>YOLOv8 Training</span>
          <div className="ripple"></div>
        </div>
        <div className="arch-connector">
          <div className="particle agent-2"></div>
        </div>
        <div className="arch-node">
          <div className="arch-icon"><BarChart3 size={20} /></div>
          <span>Evaluation</span>
          <div className="ripple"></div>
        </div>
      </div>
      <div className="arch-caption">
        <Sparkles size={14} className="sparkle-icon" /> 
        Multi-agent pipeline processing in real-time
      </div>
    </div>
  );
}

export default function App() {
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [showLabels, setShowLabels] = useState(true);
  const [panel, setPanel] = useState<"annotations" | "metrics" | "export">("annotations");
  const [phase, setPhase] = useState(0);
  const [aiReady, setAiReady] = useState<boolean | null>(null);
  const [aiModel, setAiModel] = useState("gpt-5.6-luna");
  const [aiLoading, setAiLoading] = useState(false);
  const [aiError, setAiError] = useState("");
  const [aiResults, setAiResults] = useState<Record<number, AiResult>>({});
  const [aiOverlay, setAiOverlay] = useState(false);
  const frames = demo.frames;
  const frame = frames[frameIndex];
  const useAiBoxes = aiOverlay && Boolean(frame && aiResults[frame.frame]);
  const activeObjects = useAiBoxes ? aiResults[frame.frame].objects : frame?.objects ?? [];
  const classCounts = {
    fruit: activeObjects.filter((object) => object.class === "fruit").length,
    bomb: activeObjects.filter((object) => object.class === "bomb").length,
  };
  useEffect(() => {
    let cancelled = false;
    fetch("/api/label")
      .then(async (response) => response.ok ? response.json() : null)
      .then((status) => {
        if (cancelled) return;
        setAiReady(Boolean(status?.configured));
        if (status?.model) setAiModel(status.model);
      })
      .catch(() => { if (!cancelled) setAiReady(false); });
    return () => { cancelled = true; };
  }, []);
  useEffect(() => {
    if (!playing || frames.length < 2) return;
    const timer = window.setInterval(() => setFrameIndex((current) => (current + 1) % frames.length), 250);
    return () => window.clearInterval(timer);
  }, [playing, frames.length]);

  async function analyzeFrame() {
    if (!frame || aiLoading || !aiReady) return;
    if (aiResults[frame.frame]) {
      setAiOverlay(true);
      return;
    }
    setAiLoading(true);
    setAiError("");
    setPlaying(false);
    
    try {
      const response = await fetch("/api/label", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame: frame.frame }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Could not analyze this frame");
      if (!Array.isArray(result.objects)) throw new Error("Invalid analysis response");
      setAiResults((current) => ({ ...current, [result.frame]: result as AiResult }));
      setAiOverlay(true);
    } catch (error) {
      setAiError(error instanceof Error ? error.message : "Could not analyze this frame");
    } finally {
      setAiLoading(false);
    }
  }

  const active = phases[phase];
  const ActiveIcon = active.icon;
  return (
    <div className="app-shell" id="top">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Clip2Detect home">
          <span className="brand-mark"><span /></span>
          <span><strong>Clip2Detect</strong><small>VIDEO TO YOLO DATA</small></span>
        </a>
        <nav aria-label="Main navigation">
          <a className="is-active" href="#overview">Overview</a>
          <a href="#dataset">Dataset</a>
          <a href="#evaluation">Evaluation</a>
          <a href="#docs">Docs</a>
        </nav>
        <div className="topbar-right">
          <span>Python&nbsp; / &nbsp;Open source</span>
          <a aria-label="Clip2Detect repository" href="https://github.com/9059Rohith/Clip2Detect" target="_blank" rel="noreferrer"><Code2 size={20} /></a>
        </div>
      </header>

      <main>
        <section className="intro" id="overview">
          <div>
            <p className="intro-path">GAMEPLAY <ArrowRight size={14} /> DATASET <ArrowRight size={14} /> DETECTOR</p>
            <h1>From video to detector<span>.</span></h1>
            <p className="intro-copy">Explore a real, reproducible Clip2Detect run: original gameplay frames, exact annotations, augmented data, trained weights, and measured evaluation.</p>
          </div>
          <div className="intro-aside">
            <strong>Collect. Label. Augment. Train. Evaluate.</strong>
            <p>An original Python workflow behind object detection training data.</p>
            <a href="#dataset">Explore the run <ArrowDownRight size={18} /></a>
          </div>
        </section>

        <section className="pipeline" aria-label="Clip2Detect pipeline">
          {phases.map((item, index) => {
            const Icon = item.icon;
            return (
              <button className={`pipeline-step ${index === phase ? "is-selected" : ""}`} key={item.name} onClick={() => setPhase(index)} aria-pressed={index === phase}>
                <span className="step-number">{index + 1}</span>
                <span className="step-copy"><strong>{item.name}</strong><small>{item.detail}</small></span>
                <Icon className="step-icon" size={18} />
              </button>
            );
          })}
        </section>

        <section className="architecture-section">
          <AnimatedArchitecture />
        </section>

        <section className="workbench" id="dataset" aria-label="Verified dataset explorer">
          <aside className="frame-sidebar">
            <div className="side-head"><span>Frames <b>({frames.length})</b></span><Film size={18} /></div>
            <div className="frame-list">
              {frames.map((item, index) => (
                <button key={item.frame} className={`frame-row ${index === frameIndex ? "is-current" : ""}`} onClick={() => { setFrameIndex(index); setPlaying(false); }} aria-label={`Show frame ${item.frame}`} aria-current={index === frameIndex ? "true" : undefined}>
                  <img src={`/sample/raw/${frameFile(item.frame)}`} alt="" loading={index < 7 ? "eager" : "lazy"} />
                  <span><strong>{String(item.frame).padStart(6, "0")}</strong><small>00:{String(Math.floor(index / 4)).padStart(2, "0")}.{String((index % 4) * 25).padStart(2, "0")}</small></span>
                  <ChevronRight size={16} />
                </button>
              ))}
            </div>
          </aside>

          <div className="viewer-column">
            <div className="viewer-meta"><span>FRUIT CATCH / VERIFIED FIXTURE</span><span>FRAME {String(frame?.frame ?? 0).padStart(6, "0")} / {frames.length} &nbsp;•&nbsp; 640 × 360</span></div>
            <div className="viewer-stage">
              {frame ? <img className="viewer-image" src={`/sample/raw/${frameFile(frame.frame)}`} alt={`Fruit Catch frame ${frame.frame}`} /> : <div className="viewer-empty">Dataset preparation in progress</div>}
              {showLabels && activeObjects.map((object, index) => {
                const [x1, y1, x2, y2] = object.xyxy;
                return <div key={`${frame?.frame}-${index}`} className={`annotation-box ${useAiBoxes ? "is-ai" : ""}`} style={{ left: `${x1 / 640 * 100}%`, top: `${y1 / 360 * 100}%`, width: `${(x2 - x1) / 640 * 100}%`, height: `${(y2 - y1) / 360 * 100}%`, borderColor: classColors[object.class] }}><span style={{ background: classColors[object.class] }}>{object.class}</span></div>;
              })}
              <div className="viewer-stamp">SOURCE FRAME · {useAiBoxes ? "LIVE OPENAI BOXES" : "GROUND TRUTH"} {showLabels ? "ON" : "OFF"}</div>
            </div>
            <div className="player-bar">
              <button onClick={() => setPlaying(!playing)} aria-label={playing ? "Pause frames" : "Play frames"} disabled={!frames.length}>{playing ? <Pause fill="currentColor" size={19} /> : <Play fill="currentColor" size={19} />}</button>
              <input type="range" min="0" max={Math.max(frames.length - 1, 0)} value={frameIndex} onChange={(event) => { setFrameIndex(Number(event.target.value)); setPlaying(false); }} aria-label="Scrub through video frames" />
              <span>{`00:${String(Math.floor(frameIndex / 4)).padStart(2, "0")}`} / 00:08</span>
              <button className="eye-control" onClick={() => setShowLabels(!showLabels)} aria-label={showLabels ? "Hide labels" : "Show labels"} title={showLabels ? "Hide boxes" : "Show boxes"}>{showLabels ? <Eye size={19} /> : <EyeOff size={19} />}</button>
            </div>
          </div>

          <aside className="inspector" id="evaluation">
            <div className="inspector-tabs" role="tablist" aria-label="Dataset details">
              {(["annotations", "metrics", "export"] as const).map((item) => <button key={item} role="tab" aria-selected={panel === item} className={panel === item ? "is-active" : ""} onClick={() => setPanel(item)}>{item}</button>)}
            </div>
            {panel === "annotations" && <div className="inspector-body">
              <div className="inspector-title"><span>{useAiBoxes ? "AI boxes in frame" : "Labels in frame"}</span><strong>{activeObjects.length}</strong></div>
              <div className="label-row"><i style={{ background: classColors.fruit }} /><span>fruit</span><strong>{classCounts.fruit}</strong><Eye size={16} /></div>
              <div className="label-row"><i style={{ background: classColors.bomb }} /><span>bomb</span><strong>{classCounts.bomb}</strong><Eye size={16} /></div>
              <div className="inspector-note"><Check size={17} /><span>{useAiBoxes ? `These boxes were returned by ${aiModel} through the live API.` : "Boxes come from the fixture geometry, then are encoded in YOLO format."}</span></div>
              <div className="ai-panel">
                <div className="ai-panel-heading"><Sparkles size={17} /><strong>Live AI labeling</strong></div>
                <p>Ask {aiModel} to locate fruit and bombs in this frame. Compare its boxes with the exact ground truth.</p>
                <div className="ai-status"><span className={aiReady ? "is-online" : ""} />{aiReady === null ? "Checking API" : aiReady ? "Server-side API ready" : "API key not configured"}</div>
                <button onClick={analyzeFrame} disabled={!aiReady || aiLoading || !frame}>{aiLoading ? "Analyzing frame…" : aiResults[frame?.frame ?? 0] ? "Show AI result" : "Analyze this frame"}<ArrowRight size={16} /></button>
                {aiResults[frame?.frame ?? 0] && <button className="ai-secondary" onClick={() => setAiOverlay(!useAiBoxes)}>{useAiBoxes ? "Show ground truth" : "Show AI boxes"}</button>}
                {aiError && <small role="alert">{aiError}</small>}
              </div>
            </div>}
            {panel === "metrics" && <div className="inspector-body metrics-panel">
              <h3>Measured model performance</h3>
              <p>Ultralytics validation on source-frame groups excluded from training.</p>
              <MetricRows data={demo.eval} />
              <div className="class-metrics">{demo.eval?.per_class.map((item) => <div key={item.class}><span>{item.class}</span><strong>{metric(item.ap50)} AP@50</strong></div>)}</div>
              <div className="inspector-note"><Check size={17} /><span>Augmented variants stay beside their source frame in the same split.</span></div>
            </div>}
            {panel === "export" && <div className="inspector-body export-panel">
              <h3>Artifacts from this run</h3>
              <p>Open the source footage, labels, evaluation JSON, or trained weights.</p>
              <a href="/sample/input.mp4" download><Film size={17} /> Input video <Download size={17} /></a>
              <a href="/sample/ground_truth.json" download><ScanSearch size={17} /> Ground truth <Download size={17} /></a>
              <a href="/sample/eval_results.json" download><BarChart3 size={17} /> Evaluation JSON <Download size={17} /></a>
              <a href="/sample/best.pt" download><Box size={17} /> Trained weights <Download size={17} /></a>
              <p className="export-footnote">Training runs locally in Python. This site is a live viewer of the verified artifacts.</p>
            </div>}
            <a className="inspector-cta" href="#docs"><TerminalSquare size={17} /> Run Clip2Detect locally <ArrowRight size={17} /></a>
          </aside>
        </section>

        <section className="phase-detail" aria-live="polite"><span className="phase-detail-icon"><ActiveIcon size={23} /></span><div><small>PIPELINE STAGE {String(phase + 1).padStart(2, "0")}</small><h2>{active.name}</h2><p>{active.text}</p></div><span className="phase-detail-ordinal">0{phase + 1} / 05</span></section>

        <section className="summary-section" id="docs">
          <div className="summary-heading"><div><h2>Built for the whole journey.</h2><p>Every figure on this page comes from one reproducible local integration run.</p></div><a href="https://github.com/9059Rohith/Clip2Detect" target="_blank" rel="noreferrer">Explore the source code <ExternalLink size={17} /></a></div>
          <div className="summary-grid">
            <div><span>01 / DATA</span><strong>{frames.length} frames</strong><p>Original gameplay footage sampled at four frames per second and labeled with {demo.objectCount} exact object boxes.</p></div>
            <div><span>02 / VARIATIONS</span><strong>{demo.augmentedCount} images</strong><p>Flip, brightness, contrast, and noise transformations, with corresponding YOLO annotations.</p></div>
            <div><span>03 / MODEL</span><strong>YOLOv8n</strong><p>{demo.trainCount} train images and {demo.validationCount} validation images; source groups never cross the split.</p></div>
            <div><span>04 / RESULT</span><strong>{metric(demo.eval?.map50)} mAP@50</strong><p>Observed validation score from the actual trained weights. Small fixtures are for integration testing, not a production benchmark.</p></div>
          </div>
          <div className="tech-strip"><div><Code2 size={24} /><span><strong>Architecture &amp; technology</strong><small>Video in. Labeled data and measured model out.</small></span></div><span>Python · OpenCV · Pillow · NumPy · Ultralytics YOLO · PyTorch · OpenAI Responses API · React</span></div>
          <div className="run-panel"><div><TerminalSquare size={23} /><h3>Reproduce this run</h3><p>The included sample is original and needs no cloud key. For your own video, supply an annotation manifest or select OpenAI labeling.</p></div><code>uv sync --locked<br />uv run python examples/fruit-catch/run_demo.py</code></div>
        </section>
      </main>
      <footer><span><span className="footer-dot" /> Clip2Detect</span><span>Original sample footage and exact synthetic ground truth</span><a href="#top">Back to top ↑</a></footer>
    </div>
  );
}

function MetricRows({ data }: { data: Metrics | null }) {
  return <div className="metric-rows">
    <div><span>mAP@50</span><strong>{metric(data?.map50)}</strong></div>
    <div><span>mAP@50–95</span><strong>{metric(data?.map50_95)}</strong></div>
    <div><span>Precision</span><strong>{metric(data?.precision)}</strong></div>
    <div><span>Recall</span><strong>{metric(data?.recall)}</strong></div>
  </div>;
}
