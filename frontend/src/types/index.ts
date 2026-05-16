export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: number[]; // [x1, y1, x2, y2]
}

export interface CountResult {
  success: boolean;
  people_count: number;
  detections: Detection[];
  message?: string;
  output_image_path?: string;
  timestamp?: string;
}

export interface HistoryItem extends CountResult {
  timestamp: string;
  imagePreview?: string;
}
