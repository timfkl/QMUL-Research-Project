"""
Motion tracking module with different tracking algorithm implementations.
Provides base utilities and specific tracker classes for various algorithms.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import zipfile
import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict


# ============================================================================
# UTILITY FUNCTIONS (Shared across all trackers)
# ============================================================================

def load_video(video_path: str, max_seconds: int = 30) -> Tuple[cv2.VideoCapture, int, int, int, float]:
    """
    Load video and extract properties.
    
    Args:
        video_path: Path to video file
        max_seconds: Maximum seconds to process
        
    Returns:
        Tuple of (cap, width, height, max_frames, fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * max_seconds)
    
    print(f"✅ Loaded: {width}x{height} @ {fps}fps → {max_frames} frames max")
    
    return cap, width, height, max_frames, fps


def detect_shi_tomasi_features(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect Shi-Tomasi features in a frame.
    
    Args:
        frame: Frame image (grayscale)
        
    Returns:
        Array of feature points or None
    """
    features = cv2.goodFeaturesToTrack(
        image=frame,
        maxCorners=100,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7
    )
    print(f"Initial features: {len(features) if features is not None else 0}")
    return features


# ============================================================================
# LUCAS-KANADE TRACKER CLASS
# ============================================================================

class LucasKanadeTracker:
    """
    Motion tracker using Lucas-Kanade optical flow with Shi-Tomasi feature detection.
    Suitable for tracking small feature point movements across frames.
    """
    
    def __init__(self):
        """Initialize Lucas-Kanade tracker."""
        pass
    
    def _get_lucas_kanade_params(self) -> Dict:
        """
        Get Lucas-Kanade optical flow parameters.
        
        Returns:
            Dict: Parameters for cv2.calcOpticalFlowPyrLK
        """
        return dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
    
    def _get_shi_tomasi_params(self) -> Dict:
        """
        Get Shi-Tomasi feature detection parameters.
        
        Returns:
            Dict: Parameters for cv2.goodFeaturesToTrack
        """
        return dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )
    
    def track(
        self,
        video_path: str,
        max_seconds: int = 30,
        show_progress: bool = True,
        show_keyframes: bool = True
    ) -> Tuple[List[np.ndarray], np.ndarray, int, float]:
        """
        Track features across video frames using Lucas-Kanade optical flow.
        
        Args:
            video_path: Path to video file
            max_seconds: Maximum seconds to process
            show_progress: Print frame progress every 10 frames
            show_keyframes: Display keyframes during tracking
            
        Returns:
            Tuple of (tracked_frames, centers, total_frames, fps)
        """
        cap, width, height, max_frames, fps = load_video(video_path, max_seconds)
        lk_params = self._get_lucas_kanade_params()
        shi_tomasi_params = self._get_shi_tomasi_params()
        
        ret, old_frame = cap.read()
        if not ret:
            raise IOError("Cannot read first frame")
        
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(
            image=old_gray,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )
        if p0 is not None:
            print(f"Initial features: {len(p0)}")
        
        mask = np.zeros_like(old_frame)
        tracked_frames = []
        centers = []
        # Using fourcc code for MP4 video
        fourcc = int(cv2.VideoWriter.fourcc(*'mp4v')) if hasattr(cv2.VideoWriter, 'fourcc') else cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
        out = cv2.VideoWriter('tracked_video.mp4', fourcc, int(fps), (width, height))
        
        frame_count = 0
        for frame_count in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if show_progress and frame_count % 10 == 0:
                print(f"Frame {frame_count}")
            
            if p0 is not None and len(p0) > 0:
                # Calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params) # type: ignore
                
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    
                    # Record center of mass
                    if len(good_new) > 0:
                        center = np.mean(good_new.astype(np.float32), axis=0)
                        centers.append(center)
                    
                    # Draw trajectories and points
                    for new, old in zip(good_new, good_old):
                        a, b = new.ravel().astype(int)
                        c, d = old.ravel().astype(int)
                        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
                        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
                    
                    out.write(cv2.add(frame, mask))
                    tracked_frames.append(frame)
                    
                    # Show keyframes
                    if show_keyframes and frame_count % 50 == 0:
                        plt.figure(figsize=(8, 6))
                        plt.imshow(cv2.cvtColor(cv2.add(frame, mask), cv2.COLOR_BGR2RGB))
                        plt.title(f"Frame {frame_count}")
                        plt.axis('off')
                        plt.show()
                    
                    # Refine features
                    p0 = good_new.reshape(-1, 1, 2)
                    if len(p0) < 20:
                        p0 = cv2.goodFeaturesToTrack(
                            image=gray,
                            maxCorners=100,
                            qualityLevel=0.01,
                            minDistance=10,
                            blockSize=7
                        )
            
            old_gray = gray.copy()
        
        cap.release()
        out.release()
        print(f"✅ Video: 'tracked_video.mp4' ({len(tracked_frames)} frames)")
        
        return tracked_frames, np.array(centers) if centers else np.array([]), frame_count, fps


# ============================================================================
# UTILITY FUNCTIONS (continued)
# ============================================================================

def calculate_statistics(centers: np.ndarray) -> Dict[str, float]:
    """
    Calculate tracking statistics from center positions.
    
    Args:
        centers: Array of center positions (N, 2)
        
    Returns:
        Dict with statistics (drift_x, drift_y, speed_avg, etc.)
    """
    if len(centers) < 2:
        return {
            'num_tracked': 0,
            'drift_x': 0.0,
            'drift_y': 0.0,
            'speed_avg': 0.0
        }
    
    deltas = np.diff(centers, axis=0)
    drift_x = float(np.std(deltas[:, 0]))
    drift_y = float(np.std(deltas[:, 1]))
    speeds = np.linalg.norm(deltas, axis=1)
    speed_avg = float(np.mean(speeds))
    
    stats = {
        'num_tracked': len(centers),
        'drift_x': drift_x,
        'drift_y': drift_y,
        'speed_avg': speed_avg
    }
    
    print(f"\n📊 STATS: {len(centers)} tracked | Drift X/Y: {drift_x:.1f}/{drift_y:.1f}px | Speed: {speed_avg:.1f}px/fr")
    
    return stats


def plot_trajectory(centers: np.ndarray) -> None:
    """
    Plot trajectory and motion analysis.
    
    Args:
        centers: Array of center positions (N, 2)
    """
    if len(centers) < 2:
        print("Not enough data points to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(centers[:, 0], centers[:, 1], 'r-', lw=2)
    ax1.set_title('Trajectory')
    ax1.grid(True)
    
    deltas = np.diff(centers, axis=0)
    ax2.plot(deltas[:, 0], label='ΔX')
    ax2.plot(deltas[:, 1], label='ΔY')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Motion/Frame')
    
    plt.tight_layout()
    plt.show()


def export_review_zip(
    centers: np.ndarray,
    tracked_frames: List[np.ndarray],
    frame_count: int,
    fps: float,
    video_path: str,
    output_file: str = 'tracking_review.zip'
) -> None:
    """
    Export tracking results to a ZIP file with CSV, JSON, video, and keyframes.
    
    Args:
        centers: Array of center positions (N, 2)
        tracked_frames: List of tracked frames
        frame_count: Total number of frames processed
        fps: Frames per second
        video_path: Original video path
        output_file: Output ZIP filename
    """
    centers = np.array(centers) if isinstance(centers, list) else centers
    
    print(f"Exporting {len(tracked_frames)} frames, {len(centers)} centers")
    
    # Create CSV
    if len(centers) > 0:
        deltas = np.diff(centers, axis=0)
        speeds = np.linalg.norm(deltas, axis=1)
        
        df = pd.DataFrame({
            'frame': range(len(centers)),
            'x': centers[:, 0],
            'y': centers[:, 1],
            'dx': np.r_[0, deltas[:, 0]],
            'dy': np.r_[0, deltas[:, 1]],
            'speed': np.r_[0, speeds]
        })
        df.to_csv('data.csv', index=False)
    
    # Create JSON summary
    stats = calculate_statistics(centers)
    summary = {
        'video': video_path,
        'frames_total': frame_count,
        'frames_tracked': len(centers),
        'fps': fps,
        'drift_x': stats['drift_x'],
        'drift_y': stats['drift_y'],
        'speed_avg': stats['speed_avg']
    }
    with open('summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save keyframes
    os.makedirs('kf', exist_ok=True)
    key_indices = sorted(set([0, len(tracked_frames) - 1] + list(range(0, len(tracked_frames), 50))))
    for i in key_indices:
        if 0 <= i < len(tracked_frames):
            cv2.imwrite(f'kf/f{i:04d}.png', tracked_frames[i])
    
    # Create ZIP archive
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists('data.csv'):
            zf.write('data.csv')
        zf.write('summary.json')
        if os.path.exists('tracked_video.mp4'):
            zf.write('tracked_video.mp4')
        for f in Path('kf').glob('*.png'):
            zf.write(str(f), f'kf/{f.name}')
    
    # Cleanup
    for f in ['data.csv', 'summary.json', 'tracked_video.mp4']:
        if os.path.exists(f):
            os.remove(f)
    
    if os.path.exists('kf'):
        for f in Path('kf').glob('*'):
            os.remove(str(f))
        os.rmdir('kf')
    
    print(f"✅ {output_file} created!")
    print(f"📦 Contains: CSV({len(centers)} rows) + JSON + video + {len(key_indices)} keyframes")


# ============================================================================
# CONVENIENCE FUNCTIONS FOR BACKWARD COMPATIBILITY
# ============================================================================

def track_lucas_kanade_optical_flow(
    video_path: str,
    max_seconds: int = 30,
    show_progress: bool = True,
    show_keyframes: bool = True
) -> Tuple[List[np.ndarray], np.ndarray, int, float]:
    """
    Convenience function to track using Lucas-Kanade without instantiating the class.
    
    Args:
        video_path: Path to video file
        max_seconds: Maximum seconds to process
        show_progress: Print frame progress every 10 frames
        show_keyframes: Display keyframes during tracking
        
    Returns:
        Tuple of (tracked_frames, centers, total_frames, fps)
    """
    tracker = LucasKanadeTracker()
    return tracker.track(video_path, max_seconds, show_progress, show_keyframes)
