import re

class CleanText:
    def __init__(self, text):
        self.text = text

    def clean_text_function(self):
        """Clean text by removing extra whitespace and special characters."""
        # removing filepaths and urls
        text = re.sub(r'[^\s]*\.(pdf|docx|txt)[^\s]*','',self.text)
        # removing copyright notices
        text = re.sub(r'©.*','',text)
        text = re.sub(r'C o p y r i g h t.*', '', text, flags=re.IGNORECASE)
        # normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # remove standalone symbols or single characters
        text = re.sub(r'\b\W\b', '', text)

        return text