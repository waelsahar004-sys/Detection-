"""
Real-time YOLO Webcam Inference
Detects objects from webcam stream using YOLOv8
"""

import argparse
import cv2
import logging
import os
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# Import configuration
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class YOLOInference:
    """YOLO Webcam Inference Class"""
    
    def __init__(self, model_name, device, confidence):
        """
        Initialize YOLO inference
        
        Args:
            model_name (str): Model name (e.g., 'yolov8n.pt')
            device (str): Device to use ('cpu', 'cuda', 'mps')
            confidence (float): Confidence threshold
        """
        self.model_name = model_name
        self.device = device
        self.confidence = confidence
        self.paused = False
        
        # Load model
        logger.info(f"Loading model: {model_name} on {device}")
        self.model = YOLO(model_name)
        self.model.to(device)
        logger.info(f"Model loaded successfully")
        
        # Create output directory
        Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
        
    def run(self):
        """Run inference on webcam stream"""
        # Open webcam
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        
        if not cap.isOpened():
            logger.error("Failed to open webcam")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.FPS)
        
        logger.info(f"Webcam opened - Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        logger.info("Press 'q' to quit, 's' to save frame, 'p' to pause/resume")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                if not self.paused:
                    # Run inference
                    results = self.model(frame, conf=self.confidence, verbose=False)
                    
                    # Draw annotations
                    annotated_frame = results[0].plot()
                    
                    # Add FPS counter
                    if config.SHOW_FPS:
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add pause indicator
                    display_frame = annotated_frame
                else:
                    display_frame = frame
                    cv2.putText(display_frame, "PAUSED", (10, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Display frame
                cv2.imshow("YOLO Webcam Inference", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):  # Quit
                    logger.info("Quitting...")
                    break
                elif key == ord('s'):  # Save frame
                    self.save_frame(display_frame, frame_count)
                    frame_count += 1
                elif key == ord('p'):  # Pause/Resume
                    self.paused = not self.paused
                    status = "PAUSED" if self.paused else "RESUMED"
                    logger.info(f"Inference {status}")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Webcam released and windows closed")
    
    def save_frame(self, frame, count):
        """Save frame to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config.OUTPUT_DIR}/detection_{timestamp}_{count}.jpg"
        
        cv2.imwrite(filename, frame)
        logger.info(f"Frame saved: {filename}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="YOLO Webcam Inference")
    parser.add_argument("--model", type=str, default=config.MODEL_NAME,
                       help="Model name (default: yolov8n.pt)")
    parser.add_argument("--device", type=str, default=config.DEVICE,
                       help="Device to use: cpu, cuda, mps (default: cpu)")
    parser.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD,
                       help="Confidence threshold (0.0-1.0, default: 0.5)")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                       help="Camera index (default: 0)")
    
    args = parser.parse_args()
    
    # Validate confidence
    if not 0.0 <= args.conf <= 1.0:
        logger.error("Confidence must be between 0.0 and 1.0")
        return
    
    # Update config camera index
    config.CAMERA_INDEX = args.camera
    
    # Run inference
    logger.info("=" * 50)
    logger.info("YOLO Webcam Inference Started")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Confidence: {args.conf}")
    logger.info("=" * 50)
    
    inference = YOLOInference(args.model, args.device, args.conf)
    inference.run()


if __name__ == "__main__":
    main()
