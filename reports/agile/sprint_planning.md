# Sprint Planning - AASD 4014 Object Detection Project
## Group 6 - Two Week Sprint

### Sprint Overview
- **Project**: Object Detection System (Person/Dog)
- **Duration**: 2 weeks (14 days)
- **Team Size**: 8 members
- **Sprint Goal**: Deliver a complete object detection web application

### Team Velocity
- **Estimated Story Points**: 89 points
- **Team Capacity**: 8 members × 14 days × 6 hours = 672 hours
- **Buffer**: 15% for meetings, reviews, testing

---

## Sprint Backlog

### Epic 1: Data Pipeline & ML Infrastructure
**Total: 34 Story Points**

#### User Story 1.1: Dataset Preparation
**As a Data Engineer, I want to prepare the Pascal VOC 2007 dataset so that the ML model can be trained on person/dog images**

- **Story Points**: 8
- **Assignee**: Athul Mathai (Data Engineer)
- **Acceptance Criteria**:
  - [ ] Download Pascal VOC 2007 dataset
  - [ ] Filter images containing only person or dog classes
  - [ ] Convert XML annotations to YOLO format
  - [ ] Create 80/20 train/validation split
  - [ ] Generate dataset statistics and visualizations
- **Tasks**:
  - Implement VOC dataset downloader
  - Create XML to YOLO conversion script
  - Generate train/val splits
  - Create dataset statistics report
- **Definition of Done**: Dataset ready with >200 instances per class

#### User Story 1.2: Model Training Pipeline
**As an ML Engineer, I want to implement a two-stage training pipeline so that the model achieves optimal performance**

- **Story Points**: 13
- **Assignee**: Anjana Jayakumar (ML Engineer)
- **Acceptance Criteria**:
  - [ ] Implement YOLOv5-based training pipeline
  - [ ] Stage 1: Freeze backbone training (10 epochs)
  - [ ] Stage 2: Full fine-tuning (40 epochs)
  - [ ] Apply data augmentation (flips, mosaic, color jitter)
  - [ ] Save best model weights
- **Tasks**:
  - Set up Ultralytics YOLO environment
  - Implement two-stage training logic
  - Configure augmentation pipeline
  - Create model checkpointing
- **Definition of Done**: Trained model with mAP@0.5 > 0.75

#### User Story 1.3: Model Evaluation
**As an ML Engineer, I want to comprehensively evaluate model performance so that we can assess project success**

- **Story Points**: 8
- **Assignee**: Anjana Jayakumar (ML Engineer)
- **Acceptance Criteria**:
  - [ ] Compute mAP@0.5 and mAP@0.5:0.95
  - [ ] Calculate precision, recall, F1-score per class
  - [ ] Generate confusion matrix and PR curves
  - [ ] Create sample detection visualizations
  - [ ] Save all metrics to JSON format
- **Tasks**:
  - Implement evaluation metrics calculation
  - Create visualization scripts
  - Generate evaluation report
- **Definition of Done**: Complete evaluation report with all metrics

#### User Story 1.4: Hyperparameter Tuning
**As an ML Engineer, I want to find optimal hyperparameters so that model performance is maximized**

- **Story Points**: 5
- **Assignee**: Anjana Jayakumar (ML Engineer)
- **Acceptance Criteria**:
  - [ ] Grid search across learning rates {0.01, 0.001, 0.0005}
  - [ ] Test batch sizes {8, 16, 32}
  - [ ] Evaluate image sizes {416, 512, 640}
  - [ ] Document best configuration
  - [ ] Create visualization of results
- **Tasks**:
  - Implement grid search framework
  - Run parameter sweep experiments
  - Analyze and visualize results
- **Definition of Done**: Optimal hyperparameters identified and documented

---

### Epic 2: Backend API Development  
**Total: 21 Story Points**

#### User Story 2.1: Object Detection API
**As a user, I want to upload images through an API so that I can get object detection results**

- **Story Points**: 13
- **Assignee**: Saranya Shaji (Software Engineer)
- **Acceptance Criteria**:
  - [ ] POST /api/detect endpoint accepting image uploads
  - [ ] Return JSON with predictions (class, score, bbox)
  - [ ] Handle various image formats (JPG, PNG)
  - [ ] Process images under 10MB
  - [ ] Response time under 2 seconds
- **Tasks**:
  - Implement FastAPI endpoint
  - Add file upload handling
  - Integrate YOLO model inference
  - Create response formatting
  - Add error handling
- **Definition of Done**: API returns accurate detection results

#### User Story 2.2: Detection History Storage
**As a user, I want my detection history saved so that I can review past results**

- **Story Points**: 8
- **Assignee**: Saranya Shaji (Software Engineer)
- **Acceptance Criteria**:
  - [ ] Store detection records in MongoDB
  - [ ] Include timestamp, predictions, counts
  - [ ] Save annotated images as base64
  - [ ] GET /api/detections endpoint
  - [ ] Pagination support for large datasets
- **Tasks**:
  - Design MongoDB schema
  - Implement detection storage
  - Create history retrieval endpoint
  - Add pagination logic
- **Definition of Done**: Users can view complete detection history

---

### Epic 3: Frontend Development
**Total: 18 Story Points**

#### User Story 3.1: Image Upload Interface
**As a user, I want an intuitive image upload interface so that I can easily submit images for detection**

- **Story Points**: 8
- **Assignee**: Ishika Fatwani (UX Designer)
- **Acceptance Criteria**:
  - [ ] Drag-and-drop file upload area
  - [ ] Image preview before detection
  - [ ] Progress indicator during upload
  - [ ] Clear error messages for invalid files
  - [ ] Responsive design for mobile devices
- **Tasks**:
  - Design upload UI components
  - Implement file handling logic
  - Add progress indicators
  - Create responsive layouts
- **Definition of Done**: Intuitive upload experience across devices

#### User Story 3.2: Results Visualization
**As a user, I want to see detection results with bounding boxes so that I can understand what was detected**

- **Story Points**: 10
- **Assignee**: Ishika Fatwani (UX Designer)
- **Acceptance Criteria**:
  - [ ] Display bounding boxes on uploaded image
  - [ ] Show class labels and confidence scores
  - [ ] Color-coded boxes (red=person, green=dog)
  - [ ] Summary statistics (person count, dog count)
  - [ ] Processing time display
- **Tasks**:
  - Implement canvas-based visualization
  - Create bounding box rendering
  - Design results summary cards
  - Add performance metrics display
- **Definition of Done**: Clear visual representation of detection results

---

### Epic 4: DevOps & Deployment
**Total: 8 Story Points**

#### User Story 4.1: Application Deployment
**As a DevOps engineer, I want to deploy the application so that users can access it reliably**

- **Story Points**: 5
- **Assignee**: Anu Sunny (DevOps & Deployment)
- **Acceptance Criteria**:
  - [ ] Backend deployed and accessible
  - [ ] Frontend served with proper routing
  - [ ] Database connectivity verified
  - [ ] Environment variables configured
  - [ ] Health check endpoints working
- **Tasks**:
  - Configure deployment environment
  - Set up monitoring
  - Verify service connectivity
  - Create deployment documentation
- **Definition of Done**: Application accessible and stable

#### User Story 4.2: Automation Scripts
**As a team member, I want automation scripts so that setup and deployment processes are streamlined**

- **Story Points**: 3
- **Assignee**: Anu Sunny (DevOps & Deployment)
- **Acceptance Criteria**:
  - [ ] Complete setup script for development
  - [ ] End-to-end automation script
  - [ ] Dependency installation automation
  - [ ] Clear documentation for all scripts
- **Tasks**:
  - Create setup automation
  - Write deployment scripts
  - Document usage instructions
- **Definition of Done**: One-command setup and deployment

---

### Epic 5: Documentation & Testing
**Total: 8 Story Points**

#### User Story 5.1: Project Documentation
**As a stakeholder, I want comprehensive documentation so that I can understand the project completely**

- **Story Points**: 5
- **Assignee**: Devikaa Dinesh (Report Writer)
- **Acceptance Criteria**:
  - [ ] Technical report with all required sections
  - [ ] API documentation with examples
  - [ ] User guide for web application
  - [ ] Team contribution documentation
  - [ ] README with setup instructions
- **Tasks**:
  - Write technical report
  - Create API documentation
  - Develop user guides
  - Document team contributions
- **Definition of Done**: Complete project documentation

#### User Story 5.2: Quality Assurance
**As a QA engineer, I want to test all functionality so that the application is reliable**

- **Story Points**: 3
- **Assignee**: Syed Mohamed Shakeel (QA Engineer)
- **Acceptance Criteria**:
  - [ ] Test all API endpoints
  - [ ] Verify frontend functionality
  - [ ] Test error handling scenarios
  - [ ] Performance testing for detection API
  - [ ] Cross-browser compatibility testing
- **Tasks**:
  - Create testing framework
  - Execute functional tests
  - Perform performance tests
  - Document test results
- **Definition of Done**: All critical functionality tested and verified

---

## Sprint Timeline

### Week 1 (Days 1-7)
**Focus: Foundation & ML Pipeline**

- **Day 1-2**: Sprint planning, environment setup
- **Day 3-4**: Dataset preparation and training pipeline
- **Day 5-6**: Model training and initial evaluation
- **Day 7**: Sprint review and retrospective

### Week 2 (Days 8-14)
**Focus: Integration & Deployment**

- **Day 8-9**: Backend API development
- **Day 10-11**: Frontend development and integration
- **Day 12-13**: Testing, documentation, deployment
- **Day 14**: Final demo preparation and delivery

---

## Risk Management

### High Priority Risks
1. **Model Training Time**: GPU availability and training duration
   - **Mitigation**: Use demo setup with pre-trained weights for development
   
2. **Dataset Size**: Pascal VOC download and processing time
   - **Mitigation**: Prepare subset for initial development

3. **Integration Issues**: Frontend-backend API compatibility
   - **Mitigation**: Early API contract definition and testing

### Medium Priority Risks
1. **Performance**: API response times under load
   - **Mitigation**: Optimize model loading and caching
   
2. **Browser Compatibility**: Canvas rendering across browsers
   - **Mitigation**: Test on multiple browsers early

## Definition of Done (DoD)
- [ ] Code reviewed by at least one team member
- [ ] Unit tests written and passing (where applicable)
- [ ] Documentation updated
- [ ] Acceptance criteria met
- [ ] QA testing completed
- [ ] Demo-ready functionality

---

**Sprint Planning Completed**: January 15, 2024
**Sprint Start**: January 16, 2024  
**Sprint End**: January 29, 2024
**Scrum Master**: Tri Thanh Alan Inder Kumar (Project Manager)