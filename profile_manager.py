import json
import os
import re
from typing import Dict, Any
from langchain_core.messages import HumanMessage

class ProfileManager:
    """
    Handles the extraction and updating of user profiles based on 
    conversation interactions.
    """
    def __init__(self, llm, template_path="profile_template.json"):
        self.llm = llm
        self.template_path = template_path

    def update_profile(self, clean_history: list, current_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the clean conversation history and updates the user profile JSON 
        according to the template.
        """
        if not os.path.exists(self.template_path):
            template = "{'industry': 'unknown', 'stage': 'unknown', 'goals': []}"
        else:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                template = f.read()

        prompt = (
            "You are a User Profile Architect. Your task is to update the user's profile "
            "based on the provided clean conversation history. Extract new facts about their business, "
            "industry, stage, or goals. Maintain existing information unless it is "
            "explicitly contradicted. Return the FULL updated JSON following the TEMPLATE."
        )
        
        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in clean_history])
        context = (
            f"TEMPLATE:\n{template}\n\n"
            f"CURRENT PROFILE:\n{json.dumps(current_profile)}\n\n"
            f"CONVERSATION HISTORY:\n{history_str}"
        )
        
        res = self.llm.invoke([HumanMessage(content=context)])
        response_text = res.content if hasattr(res, 'content') else str(res)
        
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return current_profile
        except Exception as e:
            print(f"Profile update error: {e}")
            return current_profile
