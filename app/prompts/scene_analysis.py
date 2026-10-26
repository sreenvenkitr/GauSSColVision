import json

SYSTEM_INSTRUCTION = """You are an expert 3D scene inspector and visual reasoning assistant integrated into GauSSCol, a 3D Gaussian Splatting inspection tool.

You receive one or more rendered image views of a 3D Gaussian Splatting scene.
CRITICAL RULES:
1. Intent Classification:
   - Identify whether the user is asking an informational question ("VISUAL_QUERY") or commanding object selection/highlighting/segmentation ("SELECT_OBJECT"):
     * VISUAL_QUERY: "What color is the bucket?", "Describe the objects", "How many chairs?", "Where is the crack?". Set "intent": "VISUAL_QUERY", "requires_segmentation": false.
     * SELECT_OBJECT: "Select the brown chair", "Highlight the bicycle", "Segment the red bucket", "Pick the table". Set "intent": "SELECT_OBJECT", "requires_segmentation": true.

2. Joint Multi-View Object Grounding (For SELECT_OBJECT):
   - When the user asks to SELECT an object, extract the target label and description, e.g. target: {"label": "chair", "description": "brown chair"}.
   - You MUST evaluate the target's visibility in EACH provided camera viewpoint (view_id).
   - For every view:
     * visible: true if the target object is clearly or partially visible in this view, false if occluded or out of frame.
     * bbox: normalized bounding box [x_min, y_min, x_max, y_max] tightly enclosing the target in that view (0.0 to 1.0).
     * confidence: your confidence score (0.0 to 1.0) that the target is in this bbox.
   - If the target object does not exist in the scene, return empty grounded_views with a clear explanation in answer: "I could not locate [target] in the available views."

3. Visual Reasoning & Defect Queries (For VISUAL_QUERY):
   - Ground all statements strictly in the visible rendered images.
   - For localized features or anomalies, provide normalized 2D bounding boxes in observations.

4. Strict JSON Output:
   Always respond with a single, valid JSON object following the exact schema below. Do not include markdown fences outside the JSON block.

JSON Schema:
{
  "intent": "VISUAL_QUERY or SELECT_OBJECT",
  "requires_segmentation": false,
  "target": {
    "label": "short object label (e.g. chair, bucket)",
    "description": "attribute description (e.g. brown chair, metal bucket)"
  },
  "grounded_views": [
    {
      "view_id": "exact_view_id_from_input",
      "visible": true,
      "bbox": {
        "x_min": 0.0 to 1.0,
        "y_min": 0.0 to 1.0,
        "x_max": 0.0 to 1.0,
        "y_max": 0.0 to 1.0
      },
      "confidence": 0.0 to 1.0
    }
  ],
  "answer": "Clear, concise direct answer or action summary (under 50 words).",
  "observations": [
    {
      "label": "descriptive label",
      "view_id": "exact_view_id_from_input",
      "bbox": {
        "x_min": 0.0 to 1.0,
        "y_min": 0.0 to 1.0,
        "x_max": 0.0 to 1.0,
        "y_max": 0.0 to 1.0
      },
      "confidence": 0.0 to 1.0,
      "description": "Visual evidence description."
    }
  ],
  "possible_explanations": ["explanation 1"],
  "uncertainty": "Statement of limitations or null",
  "recommended_next_view": {
    "required": false,
    "reason": "Why a new view is suggested",
    "yaw_delta_deg": 0.0,
    "pitch_delta_deg": 0.0,
    "zoom_factor": 1.0
  }
}
"""

def build_user_prompt(question: str, query_scope: str, views_meta: list, roi_box: dict = None, conversation_history: list = None) -> str:
    parts = []
    
    parts.append(f"Query Scope: {query_scope}")
    parts.append(f"User Question: {question}")
    
    if query_scope == "roi_2d":
        parts.append(
            "Note on 2D Selection: The images include a full scene context view ('context') and a high-resolution zoomed crop of the user's selected rectangular region ('roi_detail'). "
            "Focus your detailed analysis primarily on the 'roi_detail' while using the 'context' view to maintain orientation and scene relationship."
        )
        if roi_box:
            parts.append(f"Selected ROI Coordinates in context view: x_min={roi_box.get('x_min', 0):.3f}, y_min={roi_box.get('y_min', 0):.3f}, x_max={roi_box.get('x_max', 1):.3f}, y_max={roi_box.get('y_max', 1):.3f}")
            
    elif query_scope == "roi_3d":
        parts.append(
            "Note on 3D Selection: All provided images focus on the same selected 3D bounding volume from multiple viewpoints. "
            "Analyze the contents and boundaries of this 3D region across the perspectives."
        )
    elif query_scope == "scene":
        parts.append(
            "Note on Entire Scene: Multiple automatic orbit views of the scene are provided. Synthesize evidence across all angles to form a comprehensive answer."
        )
        
    if views_meta:
        views_desc = ", ".join([f"'{v.get('view_id')}' (role: {v.get('role', 'standard')})" for v in views_meta])
        parts.append(f"Available Views: {views_desc}")
        
    if conversation_history:
        history_lines = []
        for turn in conversation_history[-8:]:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            history_lines.append(f"{role}: {content}")
        joined_history = "\n".join(history_lines)
        parts.append(f"Recent Conversation History for Context:\n{joined_history}")
        
    is_select = any(w in question.lower() for w in ["select", "highlight", "segment", "isolate", "pick", "find"])
    if is_select:
        parts.append(
            "SELECTION DIRECTIVE: The user wants to SELECT an object.\n"
            "1. Set 'intent': 'SELECT_OBJECT' and 'requires_segmentation': true.\n"
            "2. Identify the target object name (e.g. 'chair', 'table', 'bucket').\n"
            "3. The camera view 'current' is the USER'S ACTIVE VIEWPORT. If the target object is visible in 'current', YOU MUST include 'current' in 'grounded_views' as the PRIMARY view.\n"
            "4. Also locate the target in ANY other orbit views where it is clearly visible and output its bounding box:\n"
            "   {\"view_id\": \"<view_id>\", \"visible\": true, \"bbox\": {\"x_min\": 0.X, \"y_min\": 0.Y, \"x_max\": 0.X, \"y_max\": 0.Y}, \"confidence\": 0.9}\n"
            "5. Also provide the corresponding bounding boxes under 'observations'."
        )

    parts.append("Provide your response as a valid JSON object matching the required schema.")
    return "\n\n".join(parts)
