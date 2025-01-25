from utils.video_scraper import scrape_videos
from utils.video_analyzer import analyze_videos
from utils.trend_analyzer import find_trends

def main():
    print("Starting AI Agent...")
    
    # Step 1: Scrape videos
    print("Scraping videos...")
    #scrape_videos()
    
    # Step 2: Analyze videos
    print("Analyzing videos...")
    transcription, transcript_path = analyze_videos()
    
    # Step 3: Find trends
    print("Finding trends...")
    find_trends(transcript_path)
    
    print("AI Agent finished!")

if __name__ == "__main__":
    main()