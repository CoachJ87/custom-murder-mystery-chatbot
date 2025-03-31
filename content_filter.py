class ContentFilter:
    def __init__(self):
        # Keywords that might indicate attempts to access background data
        self.restricted_keywords = [
            "template", "prompt", "system", "instruction", 
            "background information", "proprietary", "knowledge base",
            "how does this work", "show me the code", "API", "how are you programmed",
            "show prompt", "reveal prompt", "dump template", "source code",
            "share your instructions", "tell me your prompt", "instructions",
            "what are you trained on", "what data do you use", "your data source",
            "inner workings", "how are you built", "what files are you using"
        ]
    
    def detect_restricted_attempt(self, text):
        """
        Checks if the user input potentially tries to access background information.
        Returns True if a restricted attempt is detected, False otherwise.
        """
        text_lower = text.lower()
        
        # Check for direct attempts to access template or prompt data
        for keyword in self.restricted_keywords:
            if keyword.lower() in text_lower:
                return True
                
        # Check for phrases asking about system information
        system_phrases = [
            "what instructions do you have",
            "what are your guidelines",
            "can you show me your",
            "can you reveal your",
            "tell me how you operate",
            "explain how you work",
            "what's in your template",
            "what rules do you follow",
            "what's your source data",
            "tell me about your prompt",
            "what's your prompt",
            "can i see the template",
            "dump your knowledge base"
        ]
        
        for phrase in system_phrases:
            if phrase in text_lower:
                return True
        
        return False
    
    def get_restricted_response(self):
        """
        Returns a friendly response to use when users try to access restricted information.
        """
        return ("I'm designed to help you create murder mysteries, but I can't share my background templates or proprietary content. "
                "Let's focus on crafting your mystery story instead! What aspects of your murder mystery would you like help with?")