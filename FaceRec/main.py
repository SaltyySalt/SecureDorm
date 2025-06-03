import cv2
import os
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from bson.binary import Binary
import pickle
import gridfs
import certifi
import sys
import tempfile
import time


class FaceRecognitionSystem:
    def __init__(self, mongo_uri=None):
        """
        Initialize with enhanced connection handling and fallback
        """
        # Initialize with default parameters
        self.client = None
        self.db = None
        self.fs = None
        self.faces_collection = None
        self.recognizer_collection = None
        self.label_names = {}
        
        # Set default MongoDB URI if not provided
        if mongo_uri is None:
            mongo_uri = "mongodb+srv://admin:admin@cluster0.surm9vh.mongodb.net/Face?retryWrites=true&w=majority&appName=Cluster0&socketTimeoutMS=60000&connectTimeoutMS=30000"
        
        # Configure connection options based on URI type
        if "mongodb+srv://" in mongo_uri:
            connection_options = {
                'serverSelectionTimeoutMS': 5000,
                'tls': True,
                'tlsCAFile': certifi.where()
            }

        else:
            connection_options = {
                'directConnection': True,
                'serverSelectionTimeoutMS': 5000,
                'connectTimeoutMS': 10000
            }

        # Attempt connection to MongoDB Atlas
        try:
            print("🔄 Attempting to connect to MongoDB Atlas...")
            self.client = MongoClient(mongo_uri, **connection_options)
            self.client.admin.command('ping')
            print("✅ Successfully connected to MongoDB Atlas")
            
            # Initialize database components
            self.db = self.client.get_database()
            self.fs = gridfs.GridFS(self.db)
            self.faces_collection = self.db.faces
            self.recognizer_collection = self.db.recognizers
            
            # Create indexes
            self._create_indexes()
            
        except Exception as e:
            print(f"❌ MongoDB Atlas connection failed: {str(e)}")
            print("🔄 Attempting to connect to local MongoDB...")
            self._initialize_local_mongodb()

        # Initialize face detection components
        self._initialize_face_detection()
        
        # Initialize camera
        self._initialize_camera()
        
        # Load or train the recognizer
        self._load_or_train_recognizer()

    def _initialize_local_mongodb(self):
        """Fallback to local MongoDB"""
        try:
            self.client = MongoClient(
                "mongodb://localhost:27017/",
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000
            )
            self.client.admin.command('ping')
            print("✅ Connected to local MongoDB")
            
            self.db = self.client.face_recognition_db
            self.fs = gridfs.GridFS(self.db)
            self.faces_collection = self.db.faces
            self.recognizer_collection = self.db.recognizers
            
            self._create_indexes()
        except Exception as e:
            print(f"❌ Local MongoDB connection failed: {str(e)}")
            raise RuntimeError("Could not connect to any MongoDB instance")

    def _create_indexes(self):
        """Create necessary database indexes"""
        try:
            self.faces_collection.create_index([("name", 1)])
            self.faces_collection.create_index([("timestamp", -1)])
            self.recognizer_collection.create_index([("timestamp", -1)])
            print("✅ Database indexes created/verified")
        except Exception as e:
            print(f"⚠️ Error creating indexes: {str(e)}")

    def _initialize_face_detection(self):
        """Initialize OpenCV components"""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            self.recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=2,
                neighbors=16,
                grid_x=8,
                grid_y=8,
                threshold=85
            )
            print("✅ Face detection components initialized")
        except Exception as e:
            print(f"❌ Error initializing face detection: {str(e)}")
            raise

    def _initialize_camera(self):
        """Initialize video capture"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Cannot open camera")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print("✅ Camera initialized")
        except Exception as e:
            print(f"❌ Error initializing camera: {str(e)}")
            raise

    def _sanitize_for_mongodb(self, value):
        """Convert numpy types to MongoDB-compatible types"""
        if isinstance(value, (np.integer)):
            return int(value)
        elif isinstance(value, (np.floating)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            return {k: self._sanitize_for_mongodb(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._sanitize_for_mongodb(v) for v in value]
        return value

    def _safe_file_cleanup(self, file_path, max_retries=5):
        """Safely delete a file with retry logic for Windows file locking issues"""
        for attempt in range(max_retries):
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    return True
            except PermissionError as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ File cleanup attempt {attempt + 1} failed, retrying...")
                    time.sleep(0.1)  # Wait 100ms before retry
                else:
                    print(f"⚠️ Could not delete temporary file after {max_retries} attempts: {str(e)}")
                    return False
            except Exception as e:
                print(f"⚠️ Unexpected error during file cleanup: {str(e)}")
                return False
        return False

    def _save_model_to_gridfs(self, model_data, label_names):
        """Save large model file using GridFS"""
        try:
            # Create metadata document
            metadata = {
                "label_names": label_names,
                "num_samples": len(label_names),
                "training_date": datetime.now(),
                "model_type": "LBPH",
                "version": "1.0"
            }
            
            # Save model to GridFS
            model_id = self.fs.put(
                model_data,
                filename=f"face_model_{datetime.now().isoformat()}.yml",
                metadata=metadata
            )
            
            # Save reference in recognizer collection (small document)
            self.recognizer_collection.insert_one({
                "timestamp": datetime.now(),
                "model_id": model_id,
                "label_names": label_names,
                "num_samples": len(label_names),
                "training_date": datetime.now(),
                "model_size_bytes": len(model_data)
            })
            
            print(f"✅ Model saved to GridFS with ID: {model_id}")
            return model_id
            
        except Exception as e:
            print(f"❌ Error saving model to GridFS: {str(e)}")
            raise

    def _load_model_from_gridfs(self):
        """Load model from GridFS"""
        try:
            # Get latest model reference
            recognizer_data = self.recognizer_collection.find_one(
                sort=[("timestamp", -1)])
            
            if not recognizer_data:
                return None
            
            # Get model data from GridFS
            model_id = recognizer_data["model_id"]
            grid_file = self.fs.get(model_id)
            model_data = grid_file.read()
            
            # Load label names
            self.label_names = recognizer_data.get("label_names", {})
            
            print(f"✅ Loaded model from GridFS (Size: {len(model_data)} bytes)")
            return model_data
            
        except Exception as e:
            print(f"❌ Error loading model from GridFS: {str(e)}")
            return None

    def _load_or_train_recognizer(self):
        """Load or train face recognizer using GridFS for large models"""
        try:
            # Try to load existing recognizer from GridFS
            model_data = self._load_model_from_gridfs()
            
            if model_data:
                # Use a unique temporary file name
                temp_dir = tempfile.gettempdir()
                temp_filename = f"face_model_{int(time.time())}_{os.getpid()}.yml"
                model_path = os.path.join(temp_dir, temp_filename)
                
                try:
                    # Write model data to temporary file
                    with open(model_path, 'wb') as f:
                        f.write(model_data)
                    
                    # Load the model
                    self.recognizer.read(model_path)
                    print(f"✅ Loaded recognizer with {len(self.label_names)} known faces")
                    return
                    
                except Exception as e:
                    print(f"⚠️ Error loading saved model: {str(e)}")
                finally:
                    # Clean up temporary file
                    self._safe_file_cleanup(model_path)
                    
            # Train new recognizer if none exists or loading failed
            self._train_new_recognizer()
            
        except Exception as e:
            print(f"⚠️ Error in recognizer initialization: {str(e)}")
            self._train_new_recognizer()

    def _train_new_recognizer(self):
        """Train new face recognizer model with GridFS storage"""
        print("🔄 Training new face recognizer...")
        faces = []
        labels = []
        label_id = 0
        self.label_names = {}
        
        try:
            for face_data in self.faces_collection.find():
                name = face_data["name"]
                image_id = face_data["image_id"]
                
                try:
                    image_data = self.fs.get(image_id).read()
                    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        img = self._preprocess_face(img)
                        # Reduced augmentation to prevent oversized models
                        augmented = self._augment_face_data_light(img)
                        
                        for aug_img in augmented:
                            faces.append(aug_img)
                            if name not in self.label_names:
                                self.label_names[name] = label_id
                                label_id += 1
                            labels.append(self.label_names[name])
                except Exception as e:
                    print(f"⚠️ Error processing face {name}: {str(e)}")
            
            if faces:
                self.recognizer.train(faces, np.array(labels))
                print(f"✅ Trained recognizer with {len(faces)} samples")
                
                # Save the trained model using GridFS with improved file handling
                temp_dir = tempfile.gettempdir()
                temp_filename = f"face_model_training_{int(time.time())}_{os.getpid()}.yml"
                model_path = os.path.join(temp_dir, temp_filename)
                
                try:
                    # Save model to temporary file
                    self.recognizer.write(model_path)
                    
                    # Read the saved model data
                    with open(model_path, 'rb') as f:
                        model_data = f.read()
                    
                    # Check model size before saving
                    model_size_mb = len(model_data) / (1024 * 1024)
                    print(f"📊 Model size: {model_size_mb:.2f} MB")
                    
                    if len(model_data) > 15 * 1024 * 1024:  # 15MB threshold
                        print("⚠️ Model is large, using GridFS storage...")
                    
                    # Save to GridFS
                    self._save_model_to_gridfs(model_data, self.label_names)
                    
                except Exception as e:
                    print(f"❌ Error saving trained model: {str(e)}")
                    raise
                finally:
                    # Clean up temporary file with retry logic
                    self._safe_file_cleanup(model_path)
            else:
                print("ℹ️ No training data available - recognizer not trained")
                
        except Exception as e:
            print(f"❌ Error training recognizer: {str(e)}")
            raise

    def _preprocess_face(self, face_img):
        """Preprocess face image for recognition"""
        try:
            if len(face_img.shape) == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            face_img = cv2.equalizeHist(face_img)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            face_img = clahe.apply(face_img)
            face_img = cv2.GaussianBlur(face_img, (3, 3), 0)
            return cv2.resize(face_img, (200, 200))
        except Exception as e:
            print(f"⚠️ Error preprocessing face: {str(e)}")
            raise

    def _augment_face_data_light(self, face_img):
        """Generate fewer augmented training samples to reduce model size"""
        augmented = [face_img, cv2.flip(face_img, 1)]
        
        try:
            # Reduced augmentation - fewer angles and brightness variations
            for angle in [-10, 10]:
                center = (face_img.shape[1]//2, face_img.shape[0]//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(face_img, M, face_img.shape)
                augmented.append(rotated)
                
            for alpha in [0.9, 1.1]:
                adjusted = cv2.convertScaleAbs(face_img, alpha=alpha)
                augmented.append(adjusted)
        except Exception as e:
            print(f"⚠️ Error augmenting data: {str(e)}")
            
        return augmented

    def _augment_face_data(self, face_img):
        """Generate augmented training samples (original heavy version)"""
        augmented = [face_img, cv2.flip(face_img, 1)]
        
        try:
            for angle in [-15, -10, -5, 5, 10, 15]:
                center = (face_img.shape[1]//2, face_img.shape[0]//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(face_img, M, face_img.shape)
                augmented.append(rotated)
                
            for alpha in [0.8, 0.9, 1.1, 1.2]:
                adjusted = cv2.convertScaleAbs(face_img, alpha=alpha)
                augmented.append(adjusted)
        except Exception as e:
            print(f"⚠️ Error augmenting data: {str(e)}")
            
        return augmented

    def _save_face_data(self, name, img_gray, face_rect):
        """Save face data to MongoDB"""
        try:
            x, y, w, h = face_rect
            w, h = int(w), int(h)  # Convert to native Python types
            
            face_img = img_gray[y:y+h, x:x+w]
            face_img = self._preprocess_face(face_img)
            
            _, img_encoded = cv2.imencode('.jpg', face_img)
            image_id = self.fs.put(img_encoded.tobytes(), 
                                 filename=f"{name}_{datetime.now().isoformat()}.jpg")
            
            face_doc = {
                "name": name,
                "image_id": image_id,
                "timestamp": datetime.now(),
                "image_size": {"width": w, "height": h},
                "features": {
                    "eyes_detected": True,
                    "preprocessed": True,
                    "lighting": "normal"
                }
            }
            
            # Sanitize document before saving
            face_doc = self._sanitize_for_mongodb(face_doc)
            
            result = self.faces_collection.insert_one(face_doc)
            self._load_or_train_recognizer()
            return result.inserted_id
            
        except Exception as e:
            print(f"❌ Error saving face data: {str(e)}")
            raise

    def _is_real_face(self, face_roi):
        """Verify if detected region contains a real face"""
        try:
            if face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
                return False
                
            eyes = self.eye_cascade.detectMultiScale(
                face_roi,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return len(eyes) >= 1
            
        except Exception as e:
            print(f"⚠️ Error in face verification: {str(e)}")
            return False

    def capture_new_face(self, samples=3):  # Reduced default samples
        """Capture multiple samples of a new face"""
        try:
            name = input("Enter name for the new face: ").strip()
            if not name:
                print("❌ Name cannot be empty")
                return
                
            print(f"Capturing {samples} samples for {name}")
            
            captured_samples = 0
            while captured_samples < samples:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, **{
                    'scaleFactor': 1.1,
                    'minNeighbors': 5,
                    'minSize': (100, 100),
                    'maxSize': (600, 600)
                })
                
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                    (x, y, w, h) = faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                    
                    if self._is_real_face(face_roi):
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(
                            frame, 
                            f"Sample {captured_samples+1}/{samples} - Press SPACE", 
                            (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 255, 0), 
                            2
                        )
                        
                        key = cv2.waitKey(1)
                        if key == ord(' '):
                            try:
                                self._save_face_data(name, gray, faces[0])
                                captured_samples += 1
                                print(f"✅ Captured sample {captured_samples}/{samples}")
                            except Exception as e:
                                print(f"❌ Error saving sample: {str(e)}")
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                else:
                    cv2.putText(
                        frame, 
                        "No face detected", 
                        (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 0, 255), 
                        2
                    )
                
                cv2.imshow(f'Capturing Samples for {name}', frame)
                if cv2.waitKey(1) == 27:
                    break
            
            cv2.destroyAllWindows()
            print(f"ℹ️ Completed capturing {captured_samples} samples for {name}")
            
        except Exception as e:
            print(f"❌ Error in face capture: {str(e)}")
            cv2.destroyAllWindows()

    def recognize_faces(self):
        """Run real-time face recognition"""
        print("Face recognition running. Press 'q' to quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, **{
                    'scaleFactor': 1.1,
                    'minNeighbors': 5,
                    'minSize': (100, 100),
                    'maxSize': (600, 600)
                })
                
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                    (x, y, w, h) = faces[0]
                    face_roi = gray[y:y+h, x:x+w]
                    
                    if self._is_real_face(face_roi):
                        face_roi = self._preprocess_face(face_roi)
                        label, confidence = self.recognizer.predict(face_roi)
                        
                        if confidence < 85:
                            name = next(
                                (k for k, v in self.label_names.items() if v == label), 
                                "Unknown"
                            )
                            confidence_text = f"{100 - confidence:.0f}%"
                            color = (0, 255, 0)
                        else:
                            name = "Unknown"
                            confidence_text = "Low confidence"
                            color = (0, 0, 255)
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(
                            frame, 
                            name, 
                            (x, y-30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, 
                            color, 
                            2
                        )
                        cv2.putText(
                            frame, 
                            confidence_text, 
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            color, 
                            1
                        )
                
                cv2.putText(
                    frame, 
                    "Press 'q' to quit", 
                    (10, frame.shape[0]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    1
                )
                
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
                    
        except Exception as e:
            print(f"❌ Error in face recognition: {str(e)}")
        finally:
            cv2.destroyAllWindows()

    def list_registered_faces(self):
        """List all registered faces with statistics"""
        print("\nRegistered Faces:")
        print("----------------")
        
        try:
            pipeline = [
                {"$group": {
                    "_id": "$name",
                    "count": {"$sum": 1},
                    "latest": {"$max": "$timestamp"},
                    "first": {"$min": "$timestamp"}
                }},
                {"$sort": {"_id": 1}}
            ]
            
            for face in self.faces_collection.aggregate(pipeline):
                print(
                    f"{face['_id']}: {face['count']} samples | "
                    f"First: {face['first'].strftime('%Y-%m-%d')} | "
                    f"Last: {face['latest'].strftime('%Y-%m-%d')}"
                )
        except Exception as e:
            print(f"❌ Error listing faces: {str(e)}")
            
        print("----------------")

    def view_face_samples(self, name):
        """View stored samples for a specific person"""
        print(f"\nViewing samples for {name}:")
        
        try:
            samples = list(self.faces_collection.find({"name": name}).sort("timestamp", -1))
            
            if not samples:
                print(f"ℹ️ No samples found for {name}")
                return
                
            print(f"Found {len(samples)} samples. Press any key to cycle through them.")
            
            for i, sample in enumerate(samples, 1):
                try:
                    img_data = self.fs.get(sample["image_id"]).read()
                    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    cv2.putText(
                        img, 
                        f"Sample {i}/{len(samples)} - {sample['timestamp'].strftime('%Y-%m-%d')}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), 
                        2
                    )
                    
                    cv2.imshow(f"Sample {i} - {name}", img)
                    if cv2.waitKey(0) == 27:
                        break
                except Exception as e:
                    print(f"⚠️ Error displaying sample {i}: {str(e)}")
                    
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"❌ Error viewing samples: {str(e)}")

    def delete_face_data(self, name):
        """Delete all data for a specific person"""
        confirm = input(f"Are you sure you want to delete all data for {name}? (y/n): ")
        if confirm.lower() != 'y':
            print("Deletion canceled")
            return
            
        try:
            samples = list(self.faces_collection.find({"name": name}))
            
            if not samples:
                print(f"ℹ️ No samples found for {name}")
                return
                
            for sample in samples:
                self.fs.delete(sample["image_id"])
                
            result = self.faces_collection.delete_many({"name": name})
            print(f"✅ Deleted {result.deleted_count} samples for {name}")
            self._load_or_train_recognizer()
        except Exception as e:
            print(f"❌ Error deleting face data: {str(e)}")

    def cleanup_old_models(self):
        """Clean up old model files from GridFS"""
        try:
            # Keep only the latest 3 models
            old_models = list(self.recognizer_collection.find().sort("timestamp", -1).skip(3))
            
            for model_ref in old_models:
                try:
                    self.fs.delete(model_ref["model_id"])
                    self.recognizer_collection.delete_one({"_id": model_ref["_id"]})
                    print(f"✅ Cleaned up old model: {model_ref['model_id']}")
                except Exception as e:
                    print(f"⚠️ Error cleaning model {model_ref['model_id']}: {str(e)}")
                    
            print(f"✅ Model cleanup completed")
        except Exception as e:
            print(f"❌ Error during model cleanup: {str(e)}")

    def run(self):
        """Main application interface"""
        try:
            while True:
                print("\n=== Face Recognition System ===")
                print("1. Start Face Recognition")
                print("2. Register New Face")
                print("3. List Registered Faces")
                print("4. View Face Samples")
                print("5. Delete Face Data")
                print("6. Cleanup Old Models")
                print("7. Exit")
                
                choice = input("Select option (1-7): ").strip()
                
                if choice == '1':
                    self.recognize_faces()
                elif choice == '2':
                    self.capture_new_face()
                elif choice == '3':
                    self.list_registered_faces()
                elif choice == '4':
                    name = input("Enter name to view samples: ").strip()
                    if name:
                        self.view_face_samples(name)
                elif choice == '5':
                    name = input("Enter name to delete: ").strip()
                    if name:
                        self.delete_face_data(name)
                elif choice == '6':
                    self.cleanup_old_models()
                elif choice == '7':
                    break
                else:
                    print("❌ Invalid option")
        except Exception as e:
            print(f"❌ Application error: {str(e)}")
        finally:
            self.cap.release()
            if self.client:
                self.client.close()
            cv2.destroyAllWindows()
            print("ℹ️ Application closed")

if __name__ == "__main__":
    try:
        # Initialize with custom URI if needed
        # system = FaceRecognitionSystem("your_connection_string_here")
        system = FaceRecognitionSystem()
        system.run()
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)
