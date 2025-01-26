import os
import googleapiclient.discovery

def scrape_videos():
    api_key = "your api key"
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
    
    request = youtube.search().list(q="dog", part="id,snippet", type="video", maxResults=10)
    response = request.execute()
    
    # Save video metadata
    os.makedirs("data/scraped_videos", exist_ok=True)
    with open("data/scraped_videos/video_titles.txt", "w") as f:
        for item in response['items']:
            f.write(f"{item['snippet']['title']}\n")
    
    print("Video data saved to data/scraped_videos/video_titles.txt")