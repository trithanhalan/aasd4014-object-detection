import { useEffect, useState } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Object Detection Component
const DetectionPage = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [detectionResult, setDetectionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDetection = async () => {
    if (!selectedFile) {
      setError("Please select an image first");
      return;
    }

    setLoading(true);
    setError(null);
    setDetectionResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await axios.post(`${API}/detect`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setDetectionResult(response.data);
    } catch (err) {
      console.error('Detection error:', err);
      setError(err.response?.data?.detail || 'Detection failed');
    } finally {
      setLoading(false);
    }
  };

  const clearResults = () => {
    setSelectedFile(null);
    setImagePreview(null);
    setDetectionResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-6xl mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Object Detection
          </h1>
          <p className="text-lg text-gray-600">
            AASD 4014 Final Project - Group 6
          </p>
          <p className="text-sm text-gray-500 mt-1">
            Person & Dog Detection using YOLOv5
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold mb-4">Upload Image</h2>
            
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="cursor-pointer flex flex-col items-center"
              >
                <svg
                  className="w-12 h-12 text-gray-400 mb-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
                <span className="text-lg font-medium text-gray-700">
                  Click to upload image
                </span>
                <span className="text-sm text-gray-500 mt-1">
                  PNG, JPG up to 10MB
                </span>
              </label>
            </div>

            {imagePreview && (
              <div className="mt-6">
                <h3 className="text-lg font-medium mb-2">Preview</h3>
                <img
                  src={imagePreview}
                  alt="Preview"
                  className="max-w-full h-64 object-contain rounded border"
                />
              </div>
            )}

            <div className="mt-6 flex gap-3">
              <button
                onClick={handleDetection}
                disabled={!selectedFile || loading}
                className="flex-1 bg-blue-600 text-white py-3 px-6 rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Detecting...
                  </span>
                ) : (
                  'Detect Objects'
                )}
              </button>
              
              <button
                onClick={clearResults}
                className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition-colors"
              >
                Clear
              </button>
            </div>

            {error && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-red-600">{error}</p>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold mb-4">Detection Results</h2>
            
            {!detectionResult && !loading && (
              <div className="text-center py-12 text-gray-500">
                <svg
                  className="w-16 h-16 mx-auto mb-4 text-gray-300"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
                <p>Upload an image to see detection results</p>
              </div>
            )}

            {detectionResult && (
              <div className="space-y-4">
                {/* Summary */}
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h3 className="font-medium text-blue-900 mb-2">Summary</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-blue-600">Objects Found:</span>
                      <span className="ml-2 font-medium">{detectionResult.predictions.length}</span>
                    </div>
                    <div>
                      <span className="text-blue-600">Processing Time:</span>
                      <span className="ml-2 font-medium">{detectionResult.processing_time_ms}ms</span>
                    </div>
                  </div>
                </div>

                {/* Detections */}
                {detectionResult.predictions.length > 0 ? (
                  <div>
                    <h3 className="font-medium mb-3">Detected Objects</h3>
                    <div className="space-y-2">
                      {detectionResult.predictions.map((pred, index) => (
                        <div
                          key={index}
                          className="flex items-center justify-between p-3 bg-gray-50 rounded border"
                        >
                          <div className="flex items-center">
                            <div className={`w-3 h-3 rounded-full mr-3 ${
                              pred.class === 'person' ? 'bg-red-500' : 'bg-green-500'
                            }`}></div>
                            <span className="font-medium capitalize">{pred.class}</span>
                          </div>
                          <div className="text-right">
                            <div className="font-medium">{(pred.score * 100).toFixed(1)}%</div>
                            <div className="text-xs text-gray-500">confidence</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <p>No persons or dogs detected in this image</p>
                  </div>
                )}

                {/* Class Counts */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-red-50 p-4 rounded-lg text-center">
                    <div className="text-2xl font-bold text-red-600">
                      {detectionResult.predictions.filter(p => p.class === 'person').length}
                    </div>
                    <div className="text-red-600 font-medium">Persons</div>
                  </div>
                  <div className="bg-green-50 p-4 rounded-lg text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {detectionResult.predictions.filter(p => p.class === 'dog').length}
                    </div>
                    <div className="text-green-600 font-medium">Dogs</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Detection History Component
const HistoryPage = () => {
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDetections();
  }, []);

  const fetchDetections = async () => {
    try {
      const response = await axios.get(`${API}/detections`);
      setDetections(response.data.detections || []);
    } catch (error) {
      console.error('Failed to fetch detections:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading detection history...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-6xl mx-auto px-4">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Detection History
          </h1>
          <p className="text-lg text-gray-600">
            Previous object detection results
          </p>
        </div>

        {detections.length === 0 ? (
          <div className="text-center py-12">
            <svg
              className="w-16 h-16 mx-auto mb-4 text-gray-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
              />
            </svg>
            <p className="text-gray-500">No detection history found</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {detections.map((detection) => (
              <div key={detection.id} className="bg-white rounded-lg shadow-md p-6">
                <div className="mb-4">
                  <h3 className="font-semibold text-lg mb-2">
                    Detection #{detection.id.slice(-8)}
                  </h3>
                  <p className="text-sm text-gray-500">
                    {new Date(detection.timestamp).toLocaleString()}
                  </p>
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Objects Found:</span>
                    <span className="font-medium">{detection.predictions.length}</span>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 bg-red-50 rounded">
                      <div className="text-xl font-bold text-red-600">
                        {detection.person_count}
                      </div>
                      <div className="text-sm text-red-600">Persons</div>
                    </div>
                    <div className="text-center p-3 bg-green-50 rounded">
                      <div className="text-xl font-bold text-green-600">
                        {detection.dog_count}
                      </div>
                      <div className="text-sm text-green-600">Dogs</div>
                    </div>
                  </div>

                  {detection.predictions.length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-700 mb-2">Detections:</h4>
                      <div className="space-y-1">
                        {detection.predictions.slice(0, 3).map((pred, idx) => (
                          <div key={idx} className="flex justify-between text-sm">
                            <span className="capitalize">{pred.class}</span>
                            <span className="text-gray-500">{(pred.score * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                        {detection.predictions.length > 3 && (
                          <div className="text-sm text-gray-500">
                            +{detection.predictions.length - 3} more...
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

const Home = () => {
  const helloWorldApi = async () => {
    try {
      const response = await axios.get(`${API}/`);
      console.log(response.data);
    } catch (e) {
      console.error(e, `errored out requesting / api`);
    }
  };

  useEffect(() => {
    helloWorldApi();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 to-purple-900">
      <div className="container mx-auto px-4 py-16">
        {/* Header */}
        <div className="text-center mb-12">
          <img 
            src="https://avatars.githubusercontent.com/in/1201222?s=120&u=2686cf91179bbafbc7a71bfbc43004cf9ae1acea&v=4"
            alt="ObjectDetectApp Logo"
            className="w-24 h-24 mx-auto mb-6 rounded-full"
          />
          <h1 className="text-5xl font-bold text-white mb-4">
            Object Detection System
          </h1>
          <p className="text-xl text-blue-200 mb-2">
            AASD 4014 Final Project - Group 6
          </p>
          <p className="text-lg text-blue-300">
            Person & Dog Detection using YOLOv5 and Transfer Learning
          </p>
        </div>

        {/* Team Info */}
        <div className="bg-white/10 backdrop-blur-md rounded-lg p-8 mb-12">
          <h2 className="text-2xl font-semibold text-white mb-6 text-center">Team Members</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-white">
            <div className="text-center">
              <p className="font-medium">Tri Thanh Alan Inder Kumar</p>
              <p className="text-sm text-blue-200">Project Manager</p>
            </div>
            <div className="text-center">
              <p className="font-medium">Athul Mathai</p>
              <p className="text-sm text-blue-200">Data Engineer</p>
            </div>
            <div className="text-center">
              <p className="font-medium">Anjana Jayakumar</p>
              <p className="text-sm text-blue-200">ML Engineer</p>
            </div>
            <div className="text-center">
              <p className="font-medium">Anu Sunny</p>
              <p className="text-sm text-blue-200">DevOps & Deployment</p>
            </div>
            <div className="text-center">
              <p className="font-medium">Devikaa Dinesh</p>
              <p className="text-sm text-blue-200">Report Writer</p>
            </div>
            <div className="text-center">
              <p className="font-medium">Saranya Shaji</p>
              <p className="text-sm text-blue-200">Software Engineer</p>
            </div>
            <div className="text-center">
              <p className="font-medium">Syed Mohamed Shakeel</p>
              <p className="text-sm text-blue-200">QA Engineer</p>
            </div>
            <div className="text-center">
              <p className="font-medium">Ishika Fatwani</p>
              <p className="text-sm text-blue-200">UX Designer</p>
            </div>
          </div>
        </div>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
          <div className="bg-white/10 backdrop-blur-md rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-4">🎯 Object Detection</h3>
            <p className="text-blue-200 mb-4">
              Upload images to detect persons and dogs using our trained YOLOv5 model with real-time inference.
            </p>
            <Link
              to="/detect"
              className="inline-block bg-blue-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-blue-700 transition-colors"
            >
              Try Detection
            </Link>
          </div>

          <div className="bg-white/10 backdrop-blur-md rounded-lg p-6">
            <h3 className="text-xl font-semibold text-white mb-4">📊 Detection History</h3>
            <p className="text-blue-200 mb-4">
              View previous detection results with detailed statistics and performance metrics.
            </p>
            <Link
              to="/history"
              className="inline-block bg-purple-600 text-white px-6 py-2 rounded-lg font-medium hover:bg-purple-700 transition-colors"
            >
              View History
            </Link>
          </div>
        </div>

        {/* Technology Stack */}
        <div className="bg-white/10 backdrop-blur-md rounded-lg p-8">
          <h2 className="text-2xl font-semibold text-white mb-6 text-center">Technology Stack</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center text-white">
            <div>
              <div className="text-3xl mb-2">🧠</div>
              <p className="font-medium">YOLOv5</p>
              <p className="text-sm text-blue-200">ML Framework</p>
            </div>
            <div>
              <div className="text-3xl mb-2">⚡</div>
              <p className="font-medium">FastAPI</p>
              <p className="text-sm text-blue-200">Backend</p>
            </div>
            <div>
              <div className="text-3xl mb-2">⚛️</div>
              <p className="font-medium">React</p>
              <p className="text-sm text-blue-200">Frontend</p>
            </div>
            <div>
              <div className="text-3xl mb-2">🍃</div>
              <p className="font-medium">MongoDB</p>
              <p className="text-sm text-blue-200">Database</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Navigation Component
const Navigation = () => {
  return (
    <nav className="bg-white shadow-sm border-b">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="flex items-center space-x-2">
            <img 
              src="https://avatars.githubusercontent.com/in/1201222?s=120&u=2686cf91179bbafbc7a71bfbc43004cf9ae1acea&v=4"
              alt="Logo"
              className="w-8 h-8 rounded"
            />
            <span className="font-bold text-xl text-gray-900">Object Detection</span>
          </Link>
          
          <div className="flex space-x-6">
            <Link
              to="/"
              className="text-gray-600 hover:text-gray-900 font-medium transition-colors"
            >
              Home
            </Link>
            <Link
              to="/detect"
              className="text-gray-600 hover:text-gray-900 font-medium transition-colors"
            >
              Detection
            </Link>
            <Link
              to="/history"
              className="text-gray-600 hover:text-gray-900 font-medium transition-colors"
            >
              History
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/detect" element={
            <>
              <Navigation />
              <DetectionPage />
            </>
          } />
          <Route path="/history" element={
            <>
              <Navigation />
              <HistoryPage />
            </>
          } />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;
