from config import Config
from groq_client import GroqClient
from data_generation import DataGenerator
from storage import DataStorage
import time

def main():
    cfg = Config()
    client = GroqClient(cfg.GROQ_API_KEY, cfg.MODEL_NAME)
    generator = DataGenerator(cfg.PROMPT_TEMPLATE)
    storage = DataStorage(cfg.OUTPUT_FILE)

    for count in range(1, cfg.NUM_ENTRIES + 1):
        try:
            prompt = generator.generate_prompt()
            response = client.generate_completion(prompt)
            entry = generator.parse_response(response)
            
            if entry:
                storage.add_entry(entry)
                print(f"Generated entry {count}/{cfg.NUM_ENTRIES}")
            else:
                print(f"Skipped invalid entry {count}")
            
            time.sleep(1)  # Rate limit protection
            
        except Exception as e:
            print(f"Critical error at entry {count}: {str(e)}")
            time.sleep(10)

    storage.final_save()
    print("Dataset generation completed!")

if __name__ == "__main__":
    main()
