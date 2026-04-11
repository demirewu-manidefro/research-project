import React, { useState } from 'react';
import axios from 'axios';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell 
} from 'recharts';
import { Search, AlertTriangle, Sparkles, Activity, Trophy, BarChart2 } from 'lucide-react';
import './index.css';

function App() {
  const [activeTab, setActiveTab] = useState('single'); // 'single' or 'compare'
  const [url, setUrl] = useState('');
  const [compareUrls, setCompareUrls] = useState('');
  
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [compareData, setCompareData] = useState(null);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!url) return;
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const response = await axios.post('http://127.0.0.1:8000/api/analyze', {
        youtube_url: url,
        max_comments: 100
      });
      setData(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || "An error occurred connecting to the intelligence engine. Please ensure the video is public.");
    } finally {
      setLoading(false);
    }
  };

  const handleCompare = async () => {
    if (!compareUrls.trim()) return;
    const urls = compareUrls.split(/[\n\s,]+/).filter(u => u.trim() !== '');
    if (urls.length < 2) {
      setError("Please provide at least 2 URLs for comparison.");
      return;
    }
    
    setLoading(true);
    setError(null);
    setCompareData(null);
    try {
      const response = await axios.post('http://127.0.0.1:8000/api/compare', {
        youtube_urls: urls,
        max_comments: 50 // less comments for speed on multiple
      });
      setCompareData(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || "An error occurred connecting to the intelligence engine.");
    } finally {
      setLoading(false);
    }
  };

  const sentimentData = data ? [
    { name: 'Positive', count: data.sentiment_breakdown.positive, color: '#10b981' },
    { name: 'Neutral', count: data.sentiment_breakdown.neutral, color: '#64748b' },
    { name: 'Negative', count: data.sentiment_breakdown.negative, color: '#ef4444' }
  ] : [];

  return (
    <div className="dashboard-container">
      <header className="header">
        <h1>YT-Analyzer AI Engine <span style={{fontSize:'1rem', color:'#6366f1'}}>v2.0</span></h1>
        <p>Real-time Amharic Sentiment & Engagement Intelligence</p>
      </header>

      <div className="tab-switcher">
        <button className={activeTab === 'single' ? 'tab active' : 'tab'} onClick={() => setActiveTab('single')}>
          <Activity size={18}/> Single Video Analysis
        </button>
        <button className={activeTab === 'compare' ? 'tab active' : 'tab'} onClick={() => setActiveTab('compare')}>
          <Trophy size={18}/> Multi-Video Ranking
        </button>
      </div>

      {activeTab === 'single' ? (
        <div className="search-section">
          <input 
            type="text" 
            className="url-input" 
            placeholder="Paste YouTube Video URL here..." 
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
          />
          <button 
            className="analyze-btn" 
            onClick={handleAnalyze} 
            disabled={loading || !url}
          >
            {loading ? <div className="loader"></div> : <><Search size={20} /> Analyze</>}
          </button>
        </div>
      ) : (
        <div className="search-section compare-section">
          <textarea 
            className="url-input" 
            rows="4"
            placeholder="Paste multiple YouTube Video URLs (one per line)..." 
            value={compareUrls}
            onChange={(e) => setCompareUrls(e.target.value)}
          />
          <button 
            className="analyze-btn" 
            onClick={handleCompare} 
            disabled={loading || !compareUrls}
          >
            {loading ? <div className="loader"></div> : <><BarChart2 size={20} /> Compare</>}
          </button>
        </div>
      )}

      {error && (
        <div className="alert-banner" style={{marginBottom: '2rem'}}>
          <AlertTriangle /> {error}
        </div>
      )}

      {/* SINGLE VIDEO DASHBOARD */}
      {activeTab === 'single' && data && (
        <div className="grid-layout">
          {data.alert && (
            <div className="alert-banner" style={{gridColumn: 'span 12', animationDelay: '0.1s'}}>
              <AlertTriangle /> {data.alert}
            </div>
          )}

          <div className="glass-panel stat-card" style={{animationDelay: '0.2s'}}>
            <h3>Virality Score</h3>
            <div className="value score">{data.virality_score}/100</div>
            <p className="stat-desc">Success Probability</p>
          </div>

          <div className="glass-panel stat-card" style={{animationDelay: '0.3s'}}>
            <h3>Total Fetched</h3>
            <div className="value">{data.total_analyzed}</div>
            <p className="stat-desc">Real Comments Analyzed</p>
          </div>

          <div className="glass-panel stat-card" style={{animationDelay: '0.4s'}}>
            <h3>Approval Rating</h3>
            <div className="value positive">
              {Math.round((data.sentiment_breakdown.positive / data.total_analyzed) * 100)}%
            </div>
            <p className="stat-desc">Positive Audience</p>
          </div>

          <div className="glass-panel chart-section" style={{animationDelay: '0.5s'}}>
            <h3 style={{marginBottom: '1rem', color: '#f8fafc', fontWeight: 600}}>Sentiment Distribution</h3>
            <ResponsiveContainer width="100%" height="90%">
              <BarChart data={sentimentData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="name" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{backgroundColor: 'rgba(15,23,42,0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px'}} />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {sentimentData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="glass-panel gemini-insight" style={{animationDelay: '0.6s'}}>
            <h3><Sparkles size={20} /> Gemini Smart Instructions</h3>
            <p style={{lineHeight: 1.8, fontSize: '0.95rem', whiteSpace: 'pre-wrap'}}>{data.gemini_summary}</p>
          </div>

          <div className="glass-panel comments-list" style={{animationDelay: '0.7s'}}>
            <h3 style={{marginBottom: '1rem', color: '#f8fafc', fontWeight: 600}}>AI Emotion Detection Sample</h3>
            {data.sample_comments.map((comment, i) => (
              <div className="comment-item" key={i}>
                <div className="comment-text">{comment.text}</div>
                <div style={{display:'flex', gap:'0.5rem', alignItems:'center'}}>
                  <div className={`badge ${comment.emotion === 'Joy' ? 'Positive' : comment.emotion === 'Anger' ? 'Negative' : 'Neutral'}`}>
                    {comment.emotion}
                  </div>
                  <div className={`badge ${comment.sentiment}`}>
                    {comment.sentiment}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* MULTI-VIDEO DASHBOARD */}
      {activeTab === 'compare' && compareData && (
        <div className="grid-layout">
          <div className="glass-panel ranking-list" style={{gridColumn: 'span 12', animationDelay: '0.1s'}}>
            <h3 style={{marginBottom: '1.5rem', color: '#f8fafc', fontWeight: 600, display:'flex', alignItems:'center', gap:'0.5rem'}}>
              <Trophy color="#f59e0b" /> Sentiment Performance Ranking
            </h3>
            <div className="ranking-grid">
              <div className="ranking-header">Rank</div>
              <div className="ranking-header">Video ID</div>
              <div className="ranking-header">Performance Index (SPI)</div>
              <div className="ranking-header">Approval (%)</div>
              
              {compareData.ranked_videos.map((video, index) => (
                <React.Fragment key={index}>
                  <div className={`ranking-cell rank-num ${index === 0 ? 'top-rank' : ''}`}>#{index + 1}</div>
                  <div className="ranking-cell"><a href={video.video_url} target="_blank" rel="noreferrer" style={{color:'#818cf8'}}>{video.video_id}</a></div>
                  <div className="ranking-cell font-bold text-primary">{video.spi_score}</div>
                  <div className={`ranking-cell font-bold ${video.positive_ratio > 0.5 ? 'text-positive' : 'text-negative'}`}>
                    {Math.round(video.positive_ratio * 100)}%
                  </div>
                </React.Fragment>
              ))}
            </div>
            
            {compareData.errors && compareData.errors.length > 0 && (
              <div style={{marginTop: '2rem'}}>
                <h4 style={{color: '#ef4444', marginBottom: '0.5rem'}}>Processing Errors</h4>
                {compareData.errors.map((err, i) => (
                  <p key={i} style={{fontSize: '0.85rem', color: '#94a3b8'}}>{err.video_url}: {err.error}</p>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
