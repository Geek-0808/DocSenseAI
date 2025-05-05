from youtube_transcript_api import YouTubeTranscriptApi

def save_transcript(video_id, output_file):
    try:
        # Get the transcript of the video
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Open the output file to save the transcript
        with open(output_file, 'w', encoding='utf-8') as file:
            for entry in transcript:
                file.write(f"{entry['text']}\n")
        
        print(f"Transcript saved to {output_file}")
    
    except Exception as e:
        print(f"Error: {e}")

# Replace with the actual video ID from the URL
video_id = 'zECoaEZRRFU'  # Extracted from YouTube URL
output_file = 'transcript.txt'  # Output file path

save_transcript(video_id, output_file)
