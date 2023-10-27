from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from language_tool_python import LanguageTool
import torch
import iso8601
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartForConditionalGeneration, BartTokenizer

app = Flask(__name__)

def error_response(message, status_code):
    return jsonify({"error": "Transcript not available"}), 404

def calculate_chunk_size(video_duration):
    if video_duration <= 600:  
        return 800
    elif video_duration <= 1800:  
        return 1200
    elif video_duration <= 3600:  
        return 1500
    else:
        return 2000  

@app.get('/summary')
def summary_api():
    print("Request accepted")
    url = request.args.get('url', '')
    video_id = extract_video_id(url)
    
    if video_id:
        transcript = get_transcript(video_id)
        if transcript:
            video_info = YouTubeTranscriptApi.get_transcript(video_id)
            video_duration = get_video_duration(video_info)
            chunk_size = calculate_chunk_size(video_duration)
            summary = get_summary(transcript, chunk_size)
            print("Summary: ", summary)
            return summary, 200
        else:
            return "Transcript not available", 404
    else:
        return "Invalid YouTube URL", 400

@app.get('/abstractive_summary')
def abstractive_summary_api():
    print("Request for abstractive summary accepted")
    url = request.args.get('url', '')
    video_id = extract_video_id(url)
    
    if video_id:
        transcript = get_transcript(video_id)
        if transcript:
            abstractive_summary = get_abstractive_summary(transcript)
            print("Abstractive Summary: ", abstractive_summary)
            return abstractive_summary, 200
        else:
            return "Transcript not available", 404
    else:
        return "Invalid YouTube URL", 400

def extract_video_id(url):
    try:
        if 'youtube.com/watch' in url:
            video_id = url.split('v=')[1].split('&')[0]
        elif 'youtu.be' in url:
            video_id = url.split('/')[-1]
        else:
            video_id = None
        return video_id
    except Exception as e:
        print(f"Error extracting video ID: {str(e)}")
        return None

def get_video_duration(video_info):
    try:
        video_duration = 0
        for entry in video_info:
            if 'start' in entry and 'duration' in entry:
                start_time = iso8601.parse_duration(entry['start'])
                duration = iso8601.parse_duration(entry['duration'])
                end_time = start_time + duration
                if end_time.total_seconds() > video_duration:
                    video_duration = end_time.total_seconds()
        return video_duration
    except Exception as e:
        print(f"Error getting video duration: {str(e)}")
        return 0

def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id) 
        transcript = ' '.join([d['text'] for d in transcript_list])
        return transcript
    except Exception as e:
        print(f"Error fetching transcript: {str(e)}")
        return None
    
def correct_grammar(text):
    tool = LanguageTool('en-US')
    corrected_text = tool.correct(text)
    return corrected_text

def get_summary(transcript, chunk_size):
    try:
        summariser = pipeline("summarization", model="t5-small", revision="d769bba")
        summary = ''
        
        for i in range(0, len(transcript), chunk_size):
            chunk = transcript[i:i + chunk_size]
            summary_text = summariser(chunk)[0]['summary_text']
            corrected_summary_text = correct_grammar(summary_text)
            summary += corrected_summary_text + ' '
        return summary
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return None
    
def get_abstractive_summary(transcript):
    try:
       
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        device = torch.device('cpu')

        
        inputs = tokenizer.encode("summarize: " + transcript, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)

        
        summary_ids = model.generate(inputs, max_length=700 , length_penalty=2.0, num_beams=4, early_stopping=True)
        abstractive_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return abstractive_summary
    except Exception as e:
        print(f"Error generating abstractive summary: {str(e)}")
        return None

if __name__ == '__main__':
    app.run()
