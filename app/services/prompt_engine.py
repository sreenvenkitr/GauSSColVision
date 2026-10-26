import logging
from typing import List, Dict, Any, Optional

from ..prompts.scene_analysis import SYSTEM_INSTRUCTION, build_user_prompt

logger = logging.getLogger("gausscol_vision.prompt_engine")

class ScenePromptEngine:
    """
    Constructs multi-view scene inspection and object grounding prompts for VLM queries.

    Why this class exists:
    VLM reasoning requires distinct prompt assembly paths depending on the inspection scope:
    1. 'roi_2d': Guides the model to focus on the zoomed crop ('roi_detail') while referencing
       the full scene ('context') for spatial grounding.
    2. 'roi_3d': Directs the model to synthesize multiple viewpoints targeting a common 3D bounding box.
    3. 'scene': Directs joint multi-view inspection across orbit viewpoints with primary focus on 'current'.
    This class centralizes prompt templating without modifying the underlying prompt semantics.
    """

    def get_system_instruction(self) -> str:
        """
        Returns the canonical system prompt defining the JSON schema, intent classification rules,
        and multi-view bounding box grounding format.
        """
        return SYSTEM_INSTRUCTION

    def build_user_prompt(
        self,
        question: str,
        query_scope: str,
        views_meta: Optional[List[Dict[str, Any]]] = None,
        roi_box: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Builds the complete user query prompt combining scope instructions, conversation context,
        and multi-view grounding directives.
        """
        return build_user_prompt(
            question=question,
            query_scope=query_scope,
            views_meta=views_meta or [],
            roi_box=roi_box,
            conversation_history=conversation_history or []
        )
