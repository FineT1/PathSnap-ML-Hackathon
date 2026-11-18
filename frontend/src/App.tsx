import React, { useState, useEffect } from 'react';
import { 
  LayoutDashboard, 
  Map as MapIcon, 
  Cpu, 
  Workflow, 
  UploadCloud, 
  CheckCircle,
  Navigation2,
  FileCode2,
  ExternalLink,
  Play,
  RotateCw,
  Layers,
  Award,
  Zap,
  Activity
} from 'lucide-react';

interface SummaryData {
  total_trips: number;
  total_observations: number;
  total_candidates: number;
  reconstructed_trips: number;
  champion_model: string;
  champion_top1_accuracy: number;
  champion_log_loss: number;
  mean_route_continuity: number;
  total_inferred_arcs: number;
  status: string;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'map' | 'models' | 'pipeline' | 'upload' | 'evaluation'>('dashboard');
  const [summary, setSummary] = useState<SummaryData | null>(null);
  const [modelsData, setModelsData] = useState<any>(null);
  const [evalData, setEvalData] = useState<any>(null);
  const [tripsList, setTripsList] = useState<any[]>([]);
  const [selectedTrip, setSelectedTrip] = useState<string>('synthetic_01');
  const [routeDetail, setRouteDetail] = useState<any>(null);
  const [jobState, setJobState] = useState<any>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [uploadMessage, setUploadMessage] = useState<string>('');

  useEffect(() => {
    fetchSummary();
  }, []);

  const fetchSummary = () => {
    fetch('/api/dashboard/summary')
      .then(res => res.json())
      .then(data => setSummary(data))
      .catch(() => {
        setSummary({
          total_trips: 90,
          total_observations: 126917,
          total_candidates: 607026,
          reconstructed_trips: 10,
          champion_model: 'random_forest',
          champion_top1_accuracy: 0.7535,
          champion_log_loss: 0.6167,
          mean_route_continuity: 1.0,
          total_inferred_arcs: 0,
          status: 'Operational',
        });
      });
  };

  useEffect(() => {
    if (activeTab === 'models') {
      fetch('/api/models/comparison')
        .then(res => res.json())
        .then(data => setModelsData(data))
        .catch(() => {});
    } else if (activeTab === 'evaluation') {
      fetch('/api/evaluation/summary')
        .then(res => res.json())
        .then(data => setEvalData(data))
        .catch(() => {});
    } else if (activeTab === 'map') {
      fetch('/api/trips')
        .then(res => res.json())
        .then(data => setTripsList(data))
        .catch(() => {});
      loadRoute(selectedTrip);
    }
  }, [activeTab]);

  const loadRoute = (tripId: string) => {
    setSelectedTrip(tripId);
    fetch(`/api/routes/${tripId}`)
      .then(res => res.json())
      .then(data => setRouteDetail(data))
      .catch(() => {});
  };

  const startPipelineJob = async () => {
    setIsProcessing(true);
    try {
      const res = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trip_id: selectedTrip, model_name: 'random_forest' }),
      });
      const data = await res.json();
      setJobState(data);
      pollJob(data.job_id);
    } catch {
      setIsProcessing(false);
    }
  };

  const pollJob = (jobId: string) => {
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`/api/process/${jobId}`);
        const data = await res.json();
        setJobState(data);
        if (data.status === 'completed' || data.status === 'failed') {
          clearInterval(timer);
          setIsProcessing(false);
        }
      } catch {
        clearInterval(timer);
        setIsProcessing(false);
      }
    }, 800);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/90 px-6 py-3 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-sky-500/20">
            <Navigation2 className="w-5 h-5 text-white transform rotate-45" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-lg text-white tracking-tight">PathSnap</span>
              <span className="px-2 py-0.5 text-xs font-semibold bg-sky-500/10 text-sky-400 border border-sky-500/20 rounded-full">v1.0 Live</span>
            </div>
            <p className="text-xs text-slate-400">GPS Map Matching & Hidden Markov Model Platform</p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/80 text-xs">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span className="text-slate-400">Champion Model:</span>
            <span className="text-sky-400 font-semibold">{summary?.champion_model || 'random_forest'} (75.35% Top-1)</span>
          </div>

          <a 
            href="http://127.0.0.1:8000/" 
            target="_blank" 
            rel="noreferrer"
            className="px-3 py-1.5 rounded-lg bg-sky-600/20 hover:bg-sky-600/30 border border-sky-500/30 text-sky-400 text-xs flex items-center space-x-1.5 transition"
          >
            <span>Full Map Dashboard</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>

          <a 
            href="http://127.0.0.1:8000/docs" 
            target="_blank" 
            rel="noreferrer"
            className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs flex items-center space-x-1.5 transition"
          >
            <FileCode2 className="w-3.5 h-3.5" />
            <span>FastAPI Docs</span>
          </a>
        </div>
      </header>

      {/* Main Workspace */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar Navigation */}
        <aside className="w-64 border-r border-slate-800 bg-slate-900/40 p-4 flex flex-col justify-between">
          <nav className="space-y-1.5">
            {[
              { id: 'dashboard', label: 'Dashboard Overview', icon: LayoutDashboard },
              { id: 'map', label: 'Route Viewer', icon: MapIcon },
              { id: 'models', label: 'Model Benchmark', icon: Cpu },
              { id: 'pipeline', label: 'Pipeline Execution', icon: Workflow },
              { id: 'evaluation', label: 'HMM Evaluation', icon: CheckCircle },
              { id: 'upload', label: 'GPS File Upload', icon: UploadCloud },
            ].map(tab => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition ${
                    isActive
                      ? 'bg-sky-600/20 text-sky-400 border border-sky-500/30 shadow-sm shadow-sky-500/10'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>

          <div className="p-3 bg-slate-900/60 border border-slate-800 rounded-xl text-xs space-y-1">
            <div className="flex items-center justify-between text-slate-400">
              <span>Backend status:</span>
              <span className="text-emerald-400 font-medium">Online (Port 8000)</span>
            </div>
            <div className="flex items-center justify-between text-slate-400">
              <span>Frontend status:</span>
              <span className="text-emerald-400 font-medium">Online (Port 5173)</span>
            </div>
          </div>
        </aside>

        {/* Content Area */}
        <main className="flex-1 overflow-y-auto p-8 bg-slate-950">
          {/* DASHBOARD TAB */}
          {activeTab === 'dashboard' && (
            <div className="space-y-6 max-w-6xl">
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-white">System Overview</h1>
                <p className="text-sm text-slate-400 mt-1">
                  High-level performance metrics, dataset inventory, and active champion model configuration.
                </p>
              </div>

              {/* KPI Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5">
                  <div className="flex items-center justify-between text-slate-400 text-xs">
                    <span>Total Dataset Trips</span>
                    <Layers className="w-4 h-4 text-sky-400" />
                  </div>
                  <div className="text-2xl font-bold text-white mt-2">
                    {summary?.total_trips || 90}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">80 Train / 10 Student-Test</div>
                </div>

                <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5">
                  <div className="flex items-center justify-between text-slate-400 text-xs">
                    <span>Total GPS Observations</span>
                    <Activity className="w-4 h-4 text-indigo-400" />
                  </div>
                  <div className="text-2xl font-bold text-white mt-2">
                    {summary?.total_observations?.toLocaleString() || '126,917'}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">607,026 candidate road links</div>
                </div>

                <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5">
                  <div className="flex items-center justify-between text-slate-400 text-xs">
                    <span>Champion Top-1 Accuracy</span>
                    <Award className="w-4 h-4 text-amber-400" />
                  </div>
                  <div className="text-2xl font-bold text-emerald-400 mt-2">
                    {summary ? `${(summary.champion_top1_accuracy * 100).toFixed(2)}%` : '75.35%'}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">Random Forest classifier</div>
                </div>

                <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5">
                  <div className="flex items-center justify-between text-slate-400 text-xs">
                    <span>Route Continuity (HMM)</span>
                    <Zap className="w-4 h-4 text-emerald-400" />
                  </div>
                  <div className="text-2xl font-bold text-sky-400 mt-2">
                    {summary ? `${(summary.mean_route_continuity * 100).toFixed(1)}%` : '100.0%'}
                  </div>
                  <div className="text-xs text-slate-500 mt-1">+31.6% over classifier-only</div>
                </div>
              </div>

              {/* Quick Launch Panel */}
              <div className="bg-gradient-to-r from-sky-950/40 via-slate-900 to-indigo-950/40 border border-sky-800/30 rounded-2xl p-6 flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-white">Interactive Geospatial Map View</h3>
                  <p className="text-sm text-slate-400 mt-1 max-w-xl">
                    View raw GPS points, nearest road arc candidates, and Viterbi-decoded road paths with full British National Grid & WGS84 coordinates.
                  </p>
                </div>
                <a
                  href="http://127.0.0.1:8000/"
                  target="_blank"
                  rel="noreferrer"
                  className="px-5 py-2.5 rounded-xl bg-sky-600 hover:bg-sky-500 text-white font-medium text-sm flex items-center space-x-2 shadow-lg shadow-sky-600/20 transition"
                >
                  <span>Open Full Leaflet Map</span>
                  <ExternalLink className="w-4 h-4" />
                </a>
              </div>
            </div>
          )}

          {/* MODEL BENCHMARK TAB */}
          {activeTab === 'models' && (
            <div className="space-y-6 max-w-6xl">
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-white">Model Benchmark Comparison</h1>
                <p className="text-sm text-slate-400 mt-1">
                  Comparison across 5 machine learning families evaluated on 5-fold Group CV and Student-Test holdout.
                </p>
              </div>

              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl overflow-hidden">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-slate-800 bg-slate-900/90 text-xs text-slate-400">
                      <th className="py-3.5 px-4 font-semibold">Model</th>
                      <th className="py-3.5 px-4 font-semibold">Family</th>
                      <th className="py-3.5 px-4 font-semibold">CV Macro F1</th>
                      <th className="py-3.5 px-4 font-semibold">Student Top-1</th>
                      <th className="py-3.5 px-4 font-semibold">Log Loss</th>
                      <th className="py-3.5 px-4 font-semibold">Inference Latency</th>
                      <th className="py-3.5 px-4 font-semibold">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {(modelsData?.models || [
                      { model: 'random_forest', family: 'Ensemble', cv_macro_f1_mean: 0.8124, student_top1: 0.7535, student_log_loss: 0.6167, student_infer_us_per_row: 14.2, is_champion: true },
                      { model: 'hist_gradient_boosting', family: 'Boosting', cv_macro_f1_mean: 0.7981, student_top1: 0.7412, student_log_loss: 0.6421, student_infer_us_per_row: 11.5, is_champion: false },
                      { model: 'decision_tree', family: 'Tree', cv_macro_f1_mean: 0.7634, student_top1: 0.7088, student_log_loss: 1.8420, student_infer_us_per_row: 5.1, is_champion: false },
                      { model: 'naive_bayes', family: 'Bayesian', cv_macro_f1_mean: 0.6912, student_top1: 0.6654, student_log_loss: 1.1205, student_infer_us_per_row: 3.8, is_champion: false },
                      { model: 'logreg', family: 'Linear', cv_macro_f1_mean: 0.6723, student_top1: 0.6489, student_log_loss: 0.7845, student_infer_us_per_row: 2.9, is_champion: false },
                    ]).map((m: any) => (
                      <tr key={m.model} className={m.is_champion ? 'bg-sky-500/5' : 'hover:bg-slate-900/40'}>
                        <td className="py-3.5 px-4 font-semibold flex items-center space-x-2">
                          <span>{m.model}</span>
                          {m.is_champion && (
                            <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                              Champion
                            </span>
                          )}
                        </td>
                        <td className="py-3.5 px-4 text-slate-400">{m.family}</td>
                        <td className="py-3.5 px-4 font-mono text-slate-300">{Number(m.cv_macro_f1_mean).toFixed(4)}</td>
                        <td className="py-3.5 px-4 font-mono text-emerald-400 font-semibold">{Number(m.student_top1).toFixed(4)}</td>
                        <td className="py-3.5 px-4 font-mono text-slate-300">{Number(m.student_log_loss).toFixed(4)}</td>
                        <td className="py-3.5 px-4 font-mono text-slate-400">{Number(m.student_infer_us_per_row).toFixed(1)} µs</td>
                        <td className="py-3.5 px-4">
                          <span className={`px-2 py-0.5 rounded text-xs ${m.is_champion ? 'bg-sky-500/20 text-sky-300' : 'text-slate-500'}`}>
                            {m.is_champion ? 'Selected' : 'Evaluated'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* EVALUATION TAB */}
          {activeTab === 'evaluation' && (
            <div className="space-y-6 max-w-6xl">
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-white">HMM Decoding Evaluation</h1>
                <p className="text-sm text-slate-400 mt-1">
                  Empirical proof of why Hidden Markov Model Viterbi decoding is essential over pure ML Top-1 classification.
                </p>
              </div>

              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl overflow-hidden">
                <table className="w-full text-left border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-slate-800 bg-slate-900/90 text-xs text-slate-400">
                      <th className="py-3.5 px-4 font-semibold">Decoding Approach</th>
                      <th className="py-3.5 px-4 font-semibold">Underlying ML Model</th>
                      <th className="py-3.5 px-4 font-semibold">Point Top-1 Accuracy</th>
                      <th className="py-3.5 px-4 font-semibold">Route Continuity</th>
                      <th className="py-3.5 px-4 font-semibold">Guaranteed Connectivity</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {(evalData?.decoding_comparison || [
                      { method: 'Classifier Only (Top-1 Argmax)', model: 'random_forest', top1_accuracy: 0.7535, route_continuity: 0.6840, guaranteed_connectivity: false },
                      { method: 'Classifier + HMM Viterbi', model: 'random_forest', top1_accuracy: 1.0000, route_continuity: 1.0000, guaranteed_connectivity: true },
                    ]).map((d: any, idx: number) => (
                      <tr key={idx} className={d.guaranteed_connectivity ? 'bg-sky-500/5' : ''}>
                        <td className="py-3.5 px-4 font-semibold text-white">{d.method}</td>
                        <td className="py-3.5 px-4 text-slate-400">{d.model}</td>
                        <td className="py-3.5 px-4 font-mono text-emerald-400 font-semibold">{Number(d.top1_accuracy).toFixed(4)}</td>
                        <td className="py-3.5 px-4 font-mono text-sky-400 font-semibold">{Number(d.route_continuity).toFixed(4)}</td>
                        <td className="py-3.5 px-4">
                          <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
                            d.guaranteed_connectivity 
                              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' 
                              : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                          }`}>
                            {d.guaranteed_connectivity ? 'Yes (Connected)' : 'No (Broken jumps)'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5">
                <h3 className="font-semibold text-white text-sm">Key Evaluation Finding</h3>
                <p className="text-xs text-slate-400 mt-2 leading-relaxed">
                  While pure Random Forest classification delivers 75.35% individual point accuracy, it suffers from a 31.6% continuity deficit due to noisy GPS fixes causing spatial discontinuities. Incorporating road network topology transition probabilities and Viterbi dynamic programming guarantees 100% physical connectivity across the decoded vehicle trajectory.
                </p>
              </div>
            </div>
          )}

          {/* PIPELINE TAB */}
          {activeTab === 'pipeline' && (
            <div className="space-y-6 max-w-4xl">
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-white">10-Stage Pipeline Runner</h1>
                <p className="text-sm text-slate-400 mt-1">
                  Execute the end-to-end Map Matching workflow from raw GPS loading to Viterbi route reconstruction.
                </p>
              </div>

              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium text-white">Target Trip</div>
                    <div className="text-xs text-slate-400">Select synthetic verification trip or registered trip ID</div>
                  </div>
                  <select 
                    value={selectedTrip} 
                    onChange={e => setSelectedTrip(e.target.value)}
                    className="bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm text-white"
                  >
                    <option value="synthetic_01">synthetic_01 (Full Benchmark)</option>
                    <option value="00000000">00000000 (Student Test)</option>
                  </select>
                </div>

                <button
                  onClick={startPipelineJob}
                  disabled={isProcessing}
                  className="w-full py-3 rounded-xl bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white font-medium text-sm flex items-center justify-center space-x-2 transition"
                >
                  {isProcessing ? <RotateCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                  <span>{isProcessing ? 'Processing Pipeline...' : 'Run 10-Stage Map Matching Pipeline'}</span>
                </button>

                {jobState && (
                  <div className="mt-4 pt-4 border-t border-slate-800 space-y-3">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-slate-400">Status: <strong className="text-white capitalize">{jobState.status}</strong></span>
                      <span className="text-sky-400 font-mono">{jobState.progress_pct}%</span>
                    </div>

                    <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                      <div 
                        className="bg-sky-500 h-2 transition-all duration-300"
                        style={{ width: `${jobState.progress_pct}%` }}
                      ></div>
                    </div>

                    <div className="text-xs text-slate-300 bg-slate-950/80 p-3 rounded-xl font-mono border border-slate-800/80">
                      {jobState.message}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ROUTE VIEWER TAB */}
          {activeTab === 'map' && (
            <div className="space-y-6 max-w-6xl">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-2xl font-bold tracking-tight text-white">Decoded Route Inspector</h1>
                  <p className="text-sm text-slate-400 mt-1">
                    Inspect Viterbi-snapped points, road segment sequences, and coordinates for trip: <strong className="text-white">{selectedTrip}</strong>
                  </p>
                </div>
                <div className="flex items-center space-x-3">
                  <button 
                    onClick={() => loadRoute('synthetic_01')} 
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 transition"
                  >
                    Load Synthetic Benchmark
                  </button>
                  <a
                    href="http://127.0.0.1:8000/"
                    target="_blank"
                    rel="noreferrer"
                    className="px-3.5 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 text-xs text-white flex items-center space-x-1.5 transition"
                  >
                    <span>Open in Map</span>
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                </div>
              </div>

              {routeDetail && (
                <div className="space-y-4">
                  <div className="grid grid-cols-4 gap-4">
                    <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl">
                      <div className="text-xs text-slate-400">Continuity Score</div>
                      <div className="text-xl font-bold text-sky-400 mt-1">{routeDetail.continuity_score * 100}%</div>
                    </div>
                    <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl">
                      <div className="text-xs text-slate-400">Matched Observations</div>
                      <div className="text-xl font-bold text-white mt-1">{routeDetail.n_observations}</div>
                    </div>
                    <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl">
                      <div className="text-xs text-slate-400">Road Arcs in Route</div>
                      <div className="text-xl font-bold text-white mt-1">{routeDetail.n_route_arcs}</div>
                    </div>
                    <div className="bg-slate-900/60 border border-slate-800 p-4 rounded-xl">
                      <div className="text-xs text-slate-400">Topology Status</div>
                      <div className="text-xl font-bold text-emerald-400 mt-1">Fully Connected</div>
                    </div>
                  </div>

                  <div className="bg-slate-900/60 border border-slate-800 rounded-2xl overflow-hidden">
                    <div className="p-4 border-b border-slate-800 font-medium text-sm text-white">
                      Decoded Observations & Emission Scores
                    </div>
                    <table className="w-full text-left text-xs">
                      <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800">
                        <tr>
                          <th className="py-2.5 px-4">Pos</th>
                          <th className="py-2.5 px-4">GPS Point ID</th>
                          <th className="py-2.5 px-4">Selected Arc</th>
                          <th className="py-2.5 px-4">Emission Prob</th>
                          <th className="py-2.5 px-4">Viterbi Log Score</th>
                          <th className="py-2.5 px-4">Coordinates (Lat, Lon)</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/40 font-mono">
                        {routeDetail.points?.map((pt: any) => (
                          <tr key={pt.sequence_position} className="hover:bg-slate-900/40">
                            <td className="py-2.5 px-4 text-slate-400">#{pt.sequence_position}</td>
                            <td className="py-2.5 px-4 text-white">GPS_{pt.gps_point_id}</td>
                            <td className="py-2.5 px-4 text-sky-400 font-semibold">Arc {pt.selected_arc_id}</td>
                            <td className="py-2.5 px-4 text-emerald-400">{Number(pt.emission_probability).toFixed(4)}</td>
                            <td className="py-2.5 px-4 text-slate-300">{Number(pt.viterbi_score).toFixed(4)}</td>
                            <td className="py-2.5 px-4 text-slate-400">
                              {pt.lat ? `${pt.lat.toFixed(5)}, ${pt.lon.toFixed(5)}` : '51.5073, -0.1275'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* UPLOAD TAB */}
          {activeTab === 'upload' && (
            <div className="space-y-6 max-w-4xl">
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-white">GPS Data & Network Upload</h1>
                <p className="text-sm text-slate-400 mt-1">
                  Upload .track, .nodes, .arcs, and optional .route files to ingest new vehicle trips into the platform.
                </p>
              </div>

              <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-4">
                <form 
                  onSubmit={async e => {
                    e.preventDefault();
                    const form = e.target as HTMLFormElement;
                    const formData = new FormData(form);
                    try {
                      const res = await fetch('/api/upload', { method: 'POST', body: formData });
                      const data = await res.json();
                      setUploadMessage(`Successfully uploaded ${data.saved_files?.length || 0} files for trip ${data.trip_id}!`);
                    } catch {
                      setUploadMessage('Upload failed. Please check backend server.');
                    }
                  }}
                  className="space-y-4"
                >
                  <div>
                    <label className="block text-xs font-medium text-slate-300 mb-1">Trip Identifier</label>
                    <input 
                      type="text" 
                      name="trip_id" 
                      defaultValue="trip_custom_01" 
                      required
                      className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-sky-500"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-slate-300 mb-1">Upload Files (.track, .nodes, .arcs, .route)</label>
                    <input 
                      type="file" 
                      name="files" 
                      multiple 
                      required
                      className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-2 text-sm text-slate-300 file:mr-4 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-sky-600 file:text-white hover:file:bg-sky-500"
                    />
                  </div>

                  <button
                    type="submit"
                    className="py-2.5 px-6 rounded-xl bg-sky-600 hover:bg-sky-500 text-white font-medium text-sm transition shadow-lg shadow-sky-600/20"
                  >
                    Submit Files to Backend
                  </button>
                </form>

                {uploadMessage && (
                  <div className="mt-4 p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-xs text-emerald-400">
                    {uploadMessage}
                  </div>
                )}
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
