from transformers import pipeline
import os
import json
from datetime import datetime
from transformers import AutoTokenizer

def read_transcript(transcript_path):
    """
    Reads and processes the transcript file for analysis.
    This function handles both plain text and JSON transcript formats.
    """
    try:
        # First, check if the file exists
        if not os.path.exists(transcript_path):
            raise FileNotFoundError(f"Transcript file not found at: {transcript_path}")

        # Determine file type from extension
        file_extension = os.path.splitext(transcript_path)[1]
        
        if file_extension == '.json':
            # Handle JSON format transcripts
            with open(transcript_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Extract text from JSON structure
                if isinstance(data, dict):
                    return data.get('text', '')
                
        else:  # Handle .txt format
            with open(transcript_path, 'r', encoding='utf-8') as file:
                return file.read()
                
    except Exception as e:
        print(f"Error reading transcript: {str(e)}")
        raise


def chunk_text(text, model_name="distilbert-base-uncased-finetuned-sst-2-english", max_length=512):
    """
    Breaks down text into chunks that the model can process, using the model's
    own tokenizer to ensure we stay within length limits.
    
    Parameters:
        text (str): The input text to be chunked
        model_name (str): Name of the model to use for tokenization
        max_length (int): Maximum length in tokens that the model can handle
    """
    # Initialize the tokenizer for our specific model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # First, split the text into sentences
    sentences = text.replace('\n', ' ').split('.')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Check the token length of this sentence
        sentence_tokens = len(tokenizer.encode(sentence))
        
        # If adding this sentence would exceed our limit
        if current_length + sentence_tokens > max_length - 10:  # Leave some padding
            if current_chunk:  # Save the current chunk if it exists
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_length += sentence_tokens
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def find_trends(transcript_path):
    """
    Analyzes transcript content for trends using sentiment analysis,
    properly handling text length limitations.
    """
    print("Initializing trend analysis...")
    
    try:
        # Read the transcript
        print(f"Reading transcript from: {transcript_path}")
        transcript_text = read_transcript(transcript_path)
        
        # Initialize the sentiment analyzer
        print("Loading sentiment analysis model...")
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        classifier = pipeline(
            "sentiment-analysis",
            model=model_name
        )
        
        # Break the transcript into properly-sized chunks
        text_chunks = chunk_text(transcript_text, model_name=model_name)
        print(f"Analyzing {len(text_chunks)} segments of text...")
        
        # Process each chunk with error handling
        analysis_results = []
        for i, chunk in enumerate(text_chunks, 1):
            try:
                print(f"Analyzing segment {i} of {len(text_chunks)}...")
                result = classifier(chunk)
                
                analysis_entry = {
                    "segment": i,
                    "text": chunk,
                    "sentiment": result[0]["label"],
                    "confidence": round(result[0]["score"] * 100, 2),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                analysis_results.append(analysis_entry)
                
            except Exception as e:
                print(f"Warning: Error processing segment {i}: {str(e)}")
                # Log the problematic chunk for debugging
                print(f"Chunk length: {len(chunk)} characters")
                continue
        
        # Save results
        os.makedirs("data/trends", exist_ok=True)
        
        # Save detailed results as JSON
        json_path = "data/trends/detailed_analysis.json"
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        print(f"Detailed analysis saved to {json_path}")
        
        # Create and save summary
        summary_path = "data/trends/analysis_summary.txt"
        with open(summary_path, "w", encoding='utf-8') as f:
            f.write("Video Content Analysis Summary\n")
            f.write("============================\n\n")
            
            if analysis_results:
                # Calculate statistics
                total_segments = len(analysis_results)
                positive_segments = sum(1 for r in analysis_results if r["sentiment"] == "POSITIVE")
                
                f.write(f"Total Segments Analyzed: {total_segments}\n")
                f.write(f"Positive Segments: {positive_segments} ({round(positive_segments/total_segments*100, 1)}%)\n")
                f.write(f"Negative Segments: {total_segments - positive_segments} "
                       f"({round((total_segments-positive_segments)/total_segments*100, 1)}%)\n\n")
            
            else:
                f.write("No segments were successfully analyzed.\n")
        
        print(f"Summary report saved to {summary_path}")
        return analysis_results
        
    except Exception as e:
        print(f"Error in trend analysis: {str(e)}")
        raise
    