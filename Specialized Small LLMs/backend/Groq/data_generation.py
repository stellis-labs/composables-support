import json
import random
import re

class DataGenerator:
    def __init__(self, template):
        self.template = template
        self.job_titles = ["Software Engineer", "Data Scientist", "Marketing Manager",
                          "Teacher", "Nurse", "Graphic Designer"]
        self.cities = ["New York", "London", "Tokyo", "Sydney", "Berlin", "Toronto"]

    def generate_prompt(self):
        placeholders = {
            "name": "<random_name>",
            "age": str(random.randint(18, 70)),
            "email": "<generated_email>",
            "city": random.choice(self.cities),
            "job_title": random.choice(self.job_titles),
            "bio": "<realistic_biography>"
        }
        return self.template.format(**placeholders)

    def parse_response(self, response):
        try:
            # Clean response and extract JSON
            json_str = re.sub(r'[\s\S]*?({.*})[\s\S]*', r'\1', response)
            data = json.loads(json_str)
            # Validate structure
            required_fields = ["Name", "Age", "Email", "City", "Job Title", "Biography"]
            return {k: data[k] for k in required_fields} if all(k in data for k in required_fields) else None
        except Exception as e:
            print(f"Parsing error: {str(e)}")
            return None