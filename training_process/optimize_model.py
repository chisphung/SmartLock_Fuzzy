#!/usr/bin/env python3
"""
Optimize YOLOv11 model for faster inference on edge devices.

This script exports the model with:
- Reduced input resolution (320x320 by default)
- INT8 quantization (optional, requires calibration images)
- NCNN format for CPU inference

Usage:
    python optimize_model.py                          # FP32, 320x320
    python optimize_model.py --int8                   # INT8, 320x320
    python optimize_model.py --imgsz 416              # FP32, 416x416
    python optimize_model.py --int8 --imgsz 256       # INT8, 256x256
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Export optimized NCNN model for edge inference"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="../edge_side/infra/weights/yolov11n.pt",
        help="Path to source weights file (.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Input image size (default: 320 for ~4x speedup)",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Enable INT8 quantization (requires ncnn tools installed)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Enable FP16 half precision (for GPU inference)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="../edge_side/infra/dataset/data.yaml",
        help="Path to data.yaml for INT8 calibration",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated based on settings)",
    )

    args = parser.parse_args()

    # Resolve paths
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = Path(__file__).parent / weights_path
    
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent / data_path

    # Generate output directory name
    if args.output_dir:
        output_name = args.output_dir
    else:
        precision = "int8" if args.int8 else ("fp16" if args.half else "fp32")
        output_name = f"yolov11n_{precision}_{args.imgsz}"

    print("=" * 60)
    print("YOLO Model Optimization for Edge Inference")
    print("=" * 60)
    print(f"Source weights: {weights_path}")
    print(f"Output name:    {output_name}")
    print(f"Input size:     {args.imgsz}x{args.imgsz}")
    print(f"INT8:           {args.int8}")
    print(f"FP16:           {args.half}")
    if args.int8:
        print(f"Calibration:    {data_path}")
        print("\n⚠️  Note: INT8 requires ncnn tools (ncnn2table, ncnn2int8)")
        print("   Install: https://github.com/Tencent/ncnn/wiki/how-to-build")
    print("=" * 60)

    # Check if weights exist
    if not weights_path.exists():
        print(f"\n❌ Error: Weights file not found: {weights_path}")
        print("\nAvailable weight files:")
        weights_dir = Path(__file__).parent.parent / "edge_side" / "infra" / "weights"
        for f in weights_dir.glob("*.pt"):
            print(f"  - {f}")
        sys.exit(1)

    # Load model
    print("\n📦 Loading model...")
    model = YOLO(str(weights_path))
    print("✅ Model loaded successfully")

    # Export to NCNN (FP32 first, INT8 requires separate step)
    print(f"\n🔧 Exporting to NCNN (imgsz={args.imgsz})...")

    export_kwargs = {
        "format": "ncnn",
        "imgsz": args.imgsz,
        "half": args.half,
        "simplify": True,
    }

    # NOTE: Ultralytics doesn't support int8 for NCNN directly
    # We export FP32 first, then use native ncnn tools for INT8

    # Perform export
    export_path = model.export(**export_kwargs)

    print(f"\n✅ Export complete!")
    print(f"📁 Output: {export_path}")

    # Move to target location
    output_dir = Path(__file__).parent.parent / "edge_side" / "infra" / "weights" / output_name
    if Path(export_path).resolve() != output_dir.resolve():
        import shutil
        if output_dir.exists():
            print(f"\n⚠️  Removing existing output: {output_dir}")
            shutil.rmtree(output_dir)
        shutil.move(export_path, output_dir)
        print(f"📁 Moved to: {output_dir}")

    # If INT8 requested, provide instructions for native ncnn quantization
    if args.int8:
        print("\n" + "=" * 60)
        print("🔢 INT8 QUANTIZATION (Manual Steps Required)")
        print("=" * 60)
        print("\nUltralytics doesn't support direct INT8 for NCNN.")
        print("Use native ncnn tools to quantize the exported FP32 model:\n")
        
        # Get image directory for calibration
        train_images = data_path.parent / "train" / "images"
        
        print("# Step 1: Create calibration image list")
        print(f"find {train_images} -name '*.jpg' | head -200 > images.txt\n")
        
        print("# Step 2: Generate calibration table")
        print(f"ncnn2table {output_dir}/model.ncnn.param {output_dir}/model.ncnn.bin \\")
        print(f"    images.txt {output_dir}/model.table \\")
        print(f"    mean=[0,0,0] norm=[0.00392,0.00392,0.00392] \\")
        print(f"    shape=[{args.imgsz},{args.imgsz},3] pixel=BGR thread=4\n")
        
        print("# Step 3: Quantize to INT8")
        print(f"ncnn2int8 {output_dir}/model.ncnn.param {output_dir}/model.ncnn.bin \\")
        print(f"    {output_dir}/model_int8.ncnn.param {output_dir}/model_int8.ncnn.bin \\")
        print(f"    {output_dir}/model.table\n")
        
        print("=" * 60)
    
    # Print usage instructions
    print("\n" + "=" * 60)
    print("🚀 USAGE")
    print("=" * 60)
    print(f"# Test inference:")
    print(f"cd edge_side/infra")
    print(f"python3 ws_server.py --weights weights/{output_name} --imgsz {args.imgsz} --display")
    print()
    print(f"# Or in Python:")
    print(f"from ultralytics import YOLO")
    print(f"model = YOLO('weights/{output_name}')")
    print(f"results = model.predict('image.jpg', imgsz={args.imgsz})")
    print("=" * 60)

    # Estimate speedup
    baseline_size = 640
    size_speedup = (baseline_size / args.imgsz) ** 2
    int8_speedup = 2.0 if args.int8 else 1.0
    total_speedup = size_speedup * int8_speedup

    print(f"\n📈 Estimated speedup vs 640x640 FP32:")
    print(f"   Resolution: {size_speedup:.1f}x")
    if args.int8:
        print(f"   INT8:       ~{int8_speedup:.1f}x (after manual quantization)")
    print(f"   Total:      ~{total_speedup:.1f}x faster")


if __name__ == "__main__":
    main()
