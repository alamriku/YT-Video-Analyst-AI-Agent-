# YouTube Video Analysis AI Agent

This project creates an intelligent agent that analyzes YouTube videos using state-of-the-art AI models. The agent transcribes video content, performs sentiment analysis, and identifies trends, making it valuable for content creators, marketers, and researchers who want to understand video content at scale.

## Project Overview

Our AI agent combines several powerful models to provide deep insights into video content:

- **Whisper Model**: OpenAI's speech recognition system that accurately transcribes audio to text
- **GPT-2 Model**: Used for natural language processing and content analysis
- **Sentiment Analysis Model**: Evaluates the emotional tone and key themes in the content

The system processes videos through several stages:
1. Audio extraction from videos
2. Speech-to-text transcription
3. Content analysis and trend identification
4. Sentiment analysis and insight generation

## Prerequisites

Before you begin, ensure you have Python 3.8 or newer installed on your system. You'll also need:

- ffmpeg (for audio processing)
- Git (for version control)
- Sufficient disk space for models and video processing

## Installation

Follow these steps to set up the project in a virtual environment:

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai_agent_yt_video_analyst.git
cd ai_agent_yt_video_analyst

# On Windows
python -m venv ai_agent_yt_video_analyst
.\ai_agent_yt_video_analyst\Scripts\activate

# On macOS/Linux
python -m venv ai_agent_yt_video_analyst
source ai_agent_yt_video_analyst/bin/activate

pip install -r requirements.txt

ai_agent_yt_video_analyst/
├── data/                    # Data storage directory
│   ├── scraped_videos/     # Downloaded video files
│   ├── transcripts/        # Generated transcriptions
│   └── trends/            # Analysis results
├── models/                 # AI model configurations
├── utils/                  # Utility functions
│   ├── video_analyzer.py   # Video processing functions
│   └── trend_analyzer.py   # Trend analysis functions
├── main.py                # Main execution script
└── requirements.txt       # Project dependencies


# On Windows
.\ai_agent_yt_video_analyst\Scripts\activate

# On macOS/Linux
source ai_agent_yt_video_analyst/bin/activate

python main.py


The program will:

Process videos in the data/scraped_videos directory
Generate transcripts in data/transcripts
Save analysis results in data/trends

Features
Our AI agent provides several key capabilities:

Video Transcription: Converts speech to text with high accuracy using Whisper
Content Analysis: Identifies key topics and themes using GPT-2
Sentiment Analysis: Evaluates emotional tone and engagement potential
Trend Detection: Recognizes patterns and emerging topics
Detailed Reporting: Generates both detailed and summary reports

Output Format
The system generates two types of output files:

JSON Reports (data/trends/detailed_analysis.json):

Detailed sentiment scores
Timestamp-aligned analysis
Topic classification


Text Summaries (data/trends/analysis_summary.txt):

Overview of key findings
Content highlights
Trend recommendations

Troubleshooting
Common issues and solutions:
# On Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# On macOS
brew install ffmpeg


CUDA/GPU Issues: If you encounter GPU-related errors, the system will automatically fall back to CPU processing.