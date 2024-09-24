import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # Initialize reviews_df from CSV and convert Timestamp column
        self.reviews_df = pd.read_csv('data/reviews.csv')
        # Ensure Timestamp column is in datetime format
        self.reviews_df['Timestamp'] = pd.to_datetime(self.reviews_df['Timestamp'], errors='coerce')

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        if environ["REQUEST_METHOD"] == "GET":
            # Parse query parameters
            query_string = environ.get("QUERY_STRING", "")
            query_params = parse_qs(query_string)
            location = query_params.get("location", [None])[0]
            start_date = query_params.get("start_date", [None])[0]
            end_date = query_params.get("end_date", [None])[0]

            # Filter reviews
            filtered_df = self.reviews_df.copy()
            if location:
                valid_locations = [
                    "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California",
                    "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
                    "El Paso, Texas", "Escondido, California", "Fresno, California",
                    "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California",
                    "Oceanside, California", "Phoenix, Arizona", "Sacramento, California",
                    "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
                ]
                if location not in valid_locations:
                    response_body = json.dumps({"error": "Invalid location"}).encode("utf-8")
                    start_response("400 Bad Request", [
                        ("Content-Type", "application/json"),
                        ("Content-Length", str(len(response_body)))
                    ])
                    return [response_body]
                filtered_df = filtered_df[filtered_df['Location'] == location]
            if start_date:
                filtered_df = filtered_df[filtered_df['Timestamp'] >= pd.to_datetime(start_date, errors='coerce')]
            if end_date:
                filtered_df = filtered_df[filtered_df['Timestamp'] <= pd.to_datetime(end_date, errors='coerce')]

            # Analyze sentiment
            filtered_df['sentiment'] = filtered_df['ReviewBody'].apply(self.analyze_sentiment)

            # Convert Timestamps and sentiments to serializable formats
            # Assuming filtered_df is your DataFrame
            filtered_df['Timestamp'] = pd.to_datetime(filtered_df['Timestamp'], errors='coerce')

            # Now you can safely use .dt accessor
            filtered_df['Timestamp'] = filtered_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')            
            result = filtered_df.to_dict(orient='records')
            for review in result:
                review['sentiment'] = review.pop('sentiment')
            result.sort(key=lambda x: x['sentiment']['compound'], reverse=True)

            response_body = json.dumps(result, indent=2).encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            # Parse form data
            content_length = int(environ.get('CONTENT_LENGTH', 0))
            post_data = environ['wsgi.input'].read(content_length).decode('utf-8')
            data = parse_qs(post_data)
            location = data.get('Location', [None])[0]
            review_body = data.get('ReviewBody', [None])[0]


            if not location or not review_body:
                response_body = json.dumps({"error": "Missing Location or ReviewBody"}).encode("utf-8")
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

            if location:
                valid_locations = [
                    "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California",
                    "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
                    "El Paso, Texas", "Escondido, California", "Fresno, California",
                    "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California",
                    "Oceanside, California", "Phoenix, Arizona", "Sacramento, California",
                    "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
                ]
                if location not in valid_locations:
                    response_body = json.dumps({"error": "Invalid location"}).encode("utf-8")
                    start_response("400 Bad Request", [
                        ("Content-Type", "application/json"),
                        ("Content-Length", str(len(response_body)))
                    ])
                    return [response_body]

            # Create a new review
            review_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            new_review = {
                "ReviewId": review_id,
                "Location": location,
                "ReviewBody": review_body,
                "Timestamp": timestamp
            }

            # Convert new review to DataFrame and append to the existing DataFrame
            new_review_df = pd.DataFrame([new_review])
            self.reviews_df = pd.concat([self.reviews_df, new_review_df], ignore_index=True)

            response_body = json.dumps(new_review, indent=2).encode("utf-8")
            start_response("201 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
