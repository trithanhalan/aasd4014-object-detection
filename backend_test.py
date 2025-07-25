#!/usr/bin/env python3
"""
AASD 4014 Object Detection Backend API Test Suite
Tests all backend endpoints for the object detection system
"""

import requests
import json
import time
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import os
from pathlib import Path

# Configuration
BACKEND_URL = "https://1fd3b7bb-1308-4b38-b779-860fdc4d5706.preview.emergentagent.com/api"
TEST_TIMEOUT = 30

class BackendTester:
    def __init__(self):
        self.backend_url = BACKEND_URL
        self.test_results = {}
        self.session = requests.Session()
        
    def log_test(self, test_name, success, message, details=None):
        """Log test results"""
        self.test_results[test_name] = {
            'success': success,
            'message': message,
            'details': details,
            'timestamp': time.time()
        }
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        if details:
            print(f"   Details: {details}")
    
    def create_test_image(self):
        """Create a simple test image with a person-like shape"""
        # Create a 640x480 RGB image
        img = Image.new('RGB', (640, 480), color='lightblue')
        
        # Draw a simple person-like shape (rectangle for body, circle for head)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Draw a simple figure
        # Head (circle)
        draw.ellipse([300, 100, 340, 140], fill='peachpuff', outline='black')
        # Body (rectangle)
        draw.rectangle([310, 140, 330, 220], fill='red', outline='black')
        # Arms
        draw.rectangle([280, 150, 310, 160], fill='peachpuff', outline='black')
        draw.rectangle([330, 150, 360, 160], fill='peachpuff', outline='black')
        # Legs
        draw.rectangle([315, 220, 320, 280], fill='blue', outline='black')
        draw.rectangle([325, 220, 330, 280], fill='blue', outline='black')
        
        # Convert to bytes
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        buffer.seek(0)
        return buffer.getvalue()
    
    def test_api_root(self):
        """Test GET /api/ endpoint"""
        try:
            response = self.session.get(f"{self.backend_url}/", timeout=TEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                expected_fields = ['message', 'group', 'version', 'endpoints']
                
                if all(field in data for field in expected_fields):
                    self.log_test("API Root Endpoint", True, 
                                "Root endpoint working correctly", 
                                f"Response: {data}")
                else:
                    self.log_test("API Root Endpoint", False, 
                                "Missing expected fields in response",
                                f"Got: {list(data.keys())}, Expected: {expected_fields}")
            else:
                self.log_test("API Root Endpoint", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("API Root Endpoint", False, f"Request failed: {str(e)}")
    
    def test_health_check(self):
        """Test GET /api/health endpoint"""
        try:
            response = self.session.get(f"{self.backend_url}/health", timeout=TEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                expected_fields = ['status', 'model', 'database', 'timestamp']
                
                if all(field in data for field in expected_fields):
                    model_loaded = data.get('model') == 'loaded'
                    db_connected = data.get('database') == 'connected'
                    
                    if model_loaded and db_connected:
                        self.log_test("Health Check Endpoint", True, 
                                    "System healthy - model loaded and DB connected", 
                                    f"Status: {data['status']}")
                    else:
                        self.log_test("Health Check Endpoint", False, 
                                    f"System not fully healthy - Model: {data.get('model')}, DB: {data.get('database')}")
                else:
                    self.log_test("Health Check Endpoint", False, 
                                "Missing expected fields in health response")
            else:
                self.log_test("Health Check Endpoint", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Health Check Endpoint", False, f"Request failed: {str(e)}")
    
    def test_object_detection(self):
        """Test POST /api/detect endpoint with image upload"""
        try:
            # Create test image
            test_image = self.create_test_image()
            
            # Prepare multipart form data
            files = {'file': ('test_image.jpg', test_image, 'image/jpeg')}
            
            response = self.session.post(f"{self.backend_url}/detect", 
                                       files=files, timeout=TEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                expected_fields = ['success', 'predictions', 'image_id', 'timestamp', 'processing_time_ms']
                
                if all(field in data for field in expected_fields):
                    if data.get('success'):
                        predictions = data.get('predictions', [])
                        self.log_test("Object Detection API", True, 
                                    f"Detection successful with {len(predictions)} predictions", 
                                    f"Processing time: {data.get('processing_time_ms')}ms, Image ID: {data.get('image_id')}")
                        
                        # Store image_id for history test
                        self.test_image_id = data.get('image_id')
                    else:
                        self.log_test("Object Detection API", False, 
                                    "Detection returned success=false")
                else:
                    self.log_test("Object Detection API", False, 
                                "Missing expected fields in detection response",
                                f"Got fields: {list(data.keys())}")
            else:
                self.log_test("Object Detection API", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Object Detection API", False, f"Request failed: {str(e)}")
    
    def test_detection_history(self):
        """Test GET /api/detections endpoint"""
        try:
            response = self.session.get(f"{self.backend_url}/detections", timeout=TEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                expected_fields = ['success', 'detections', 'count']
                
                if all(field in data for field in expected_fields):
                    if data.get('success'):
                        detections = data.get('detections', [])
                        count = data.get('count', 0)
                        
                        self.log_test("Detection History API", True, 
                                    f"Retrieved {count} detection records", 
                                    f"Sample fields in records: {list(detections[0].keys()) if detections else 'No records'}")
                    else:
                        self.log_test("Detection History API", False, 
                                    "History API returned success=false")
                else:
                    self.log_test("Detection History API", False, 
                                "Missing expected fields in history response")
            else:
                self.log_test("Detection History API", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Detection History API", False, f"Request failed: {str(e)}")
    
    def test_detection_detail(self):
        """Test GET /api/detections/{id} endpoint"""
        if not hasattr(self, 'test_image_id'):
            self.log_test("Detection Detail API", False, 
                        "No image ID available from detection test")
            return
            
        try:
            response = self.session.get(f"{self.backend_url}/detections/{self.test_image_id}", 
                                      timeout=TEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                expected_fields = ['id', 'image_id', 'timestamp', 'predictions', 'person_count', 'dog_count']
                
                if all(field in data for field in expected_fields):
                    self.log_test("Detection Detail API", True, 
                                "Retrieved detailed detection record with image data", 
                                f"Record ID: {data.get('id')}, Predictions: {len(data.get('predictions', []))}")
                else:
                    self.log_test("Detection Detail API", False, 
                                "Missing expected fields in detail response",
                                f"Got fields: {list(data.keys())}")
            elif response.status_code == 404:
                self.log_test("Detection Detail API", False, 
                            "Detection record not found - possible database storage issue")
            else:
                self.log_test("Detection Detail API", False, 
                            f"HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            self.log_test("Detection Detail API", False, f"Request failed: {str(e)}")
    
    def test_mongodb_integration(self):
        """Test MongoDB integration through detection storage and retrieval"""
        # This is tested implicitly through the detection and history endpoints
        # We'll verify by checking if data persists between calls
        
        try:
            # First, get current detection count
            response1 = self.session.get(f"{self.backend_url}/detections", timeout=TEST_TIMEOUT)
            if response1.status_code != 200:
                self.log_test("MongoDB Integration", False, "Cannot retrieve detections to test DB")
                return
                
            initial_count = response1.json().get('count', 0)
            
            # Perform a detection (which should store to DB)
            test_image = self.create_test_image()
            files = {'file': ('test_db.jpg', test_image, 'image/jpeg')}
            
            detect_response = self.session.post(f"{self.backend_url}/detect", 
                                              files=files, timeout=TEST_TIMEOUT)
            
            if detect_response.status_code != 200:
                self.log_test("MongoDB Integration", False, "Detection failed, cannot test DB storage")
                return
            
            # Wait a moment for DB write
            time.sleep(1)
            
            # Check if count increased
            response2 = self.session.get(f"{self.backend_url}/detections", timeout=TEST_TIMEOUT)
            if response2.status_code == 200:
                new_count = response2.json().get('count', 0)
                
                if new_count > initial_count:
                    self.log_test("MongoDB Integration", True, 
                                f"Database storage working - count increased from {initial_count} to {new_count}")
                else:
                    self.log_test("MongoDB Integration", False, 
                                f"Database storage may not be working - count remained {initial_count}")
            else:
                self.log_test("MongoDB Integration", False, 
                            "Cannot verify DB storage - history endpoint failed")
                
        except Exception as e:
            self.log_test("MongoDB Integration", False, f"DB integration test failed: {str(e)}")
    
    def test_yolo_model_integration(self):
        """Test YOLO model integration through health check and detection"""
        # Check model status via health endpoint
        try:
            health_response = self.session.get(f"{self.backend_url}/health", timeout=TEST_TIMEOUT)
            
            if health_response.status_code == 200:
                health_data = health_response.json()
                model_status = health_data.get('model')
                
                if model_status == 'loaded':
                    # Test actual inference
                    test_image = self.create_test_image()
                    files = {'file': ('test_model.jpg', test_image, 'image/jpeg')}
                    
                    detect_response = self.session.post(f"{self.backend_url}/detect", 
                                                      files=files, timeout=TEST_TIMEOUT)
                    
                    if detect_response.status_code == 200:
                        detect_data = detect_response.json()
                        if detect_data.get('success'):
                            processing_time = detect_data.get('processing_time_ms', 0)
                            predictions = detect_data.get('predictions', [])
                            
                            self.log_test("YOLO Model Integration", True, 
                                        f"Model loaded and inference working - {len(predictions)} predictions in {processing_time}ms")
                        else:
                            self.log_test("YOLO Model Integration", False, 
                                        "Model loaded but inference failed")
                    else:
                        self.log_test("YOLO Model Integration", False, 
                                    f"Model loaded but detection endpoint failed: {detect_response.status_code}")
                else:
                    self.log_test("YOLO Model Integration", False, 
                                f"Model not loaded - status: {model_status}")
            else:
                self.log_test("YOLO Model Integration", False, 
                            "Cannot check model status - health endpoint failed")
                
        except Exception as e:
            self.log_test("YOLO Model Integration", False, f"Model integration test failed: {str(e)}")
    
    def run_all_tests(self):
        """Run all backend tests in order"""
        print(f"🚀 Starting AASD 4014 Backend API Tests")
        print(f"Backend URL: {self.backend_url}")
        print("=" * 60)
        
        # Test in logical order
        self.test_api_root()
        self.test_health_check()
        self.test_yolo_model_integration()
        self.test_mongodb_integration()
        self.test_object_detection()
        self.test_detection_history()
        self.test_detection_detail()
        
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results.values() if result['success'])
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"{status} {test_name}")
            if not result['success']:
                print(f"   Error: {result['message']}")
        
        print(f"\nResults: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All backend tests PASSED!")
        else:
            print(f"⚠️  {total - passed} tests FAILED - see details above")
        
        return self.test_results

if __name__ == "__main__":
    tester = BackendTester()
    results = tester.run_all_tests()