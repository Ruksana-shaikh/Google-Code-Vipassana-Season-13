# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import uuid
import traceback
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.cloud import storage
from sqlalchemy import create_engine, text

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# --- Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# Initialize Clients
genai_client = None # Initialize to None
storage_client = None
engine = None #Initialize to None

try:
    genai_client = genai.Client(api_key=API_KEY)
    storage_client = storage.Client()
    
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set in environment variables.")
    
    # Increase pool size and handle disconnected sessions
    # pool_pre_ping checks if the connection is alive before using it
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
except Exception as e:
    print(f"Initialization Error: {traceback.format_exc()}")


def upload_to_gcs(file_bytes, filename):
    """Uploads a file to Google Cloud Storage and returns the public URL."""
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"items/{uuid.uuid4()}-{filename}")
    blob.upload_from_string(file_bytes, content_type="image/jpeg")
    return blob.public_url



@app.route('/')
def home():
    """
    Fetches items and renders the app.html template with the item data.
    """
    if engine is None:
        return jsonify({"error": "Database engine not initialized."}), 500

    try:
        with engine.connect() as conn:
            query = text("""
                SELECT item_id, title, bio, category, image_url 
                FROM items 
                WHERE status = 'available'
                ORDER BY created_at DESC
                LIMIT 20
            """)
            result = conn.execute(query)
            
            items = []
            for row in result:
                items.append({
                    "id": str(row[0]),
                    "title": row[1],
                    "bio": row[2],
                    "category": row[3],
                    "image_url": row[4]
                })
            
            # For SELECT, commit isn't required but keeps the connection handling consistent
            conn.commit()
            return render_template('app.html', items=items)
            
    except Exception as e:
        print(f"Error fetching items: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to fetch items", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/items', methods=['GET'])
def get_items():
    """
    Fetches available items from the database to populate the deck.
    Standardized to use the same variable-based query execution as list_item.
    """
    if engine is None:
        return jsonify({"error": "Database engine not initialized."}), 500
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT item_id, title, bio, category, image_url 
                FROM items 
                WHERE status = 'available'
                ORDER BY created_at DESC
                LIMIT 20
            """)
            result = conn.execute(query)
            
            items = []
            for row in result:
                items.append({
                    "id": str(row[0]),
                    "title": row[1],
                    "bio": row[2],
                    "category": row[3],
                    "image_url": row[4]
                })
            
            # For SELECT, commit isn't required but keeps the connection handling consistent
            conn.commit()
            return jsonify(items)
            
    except Exception as e:
        print(f"Error fetching items: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to fetch items", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/list-item', methods=['POST'])
def list_item():
    """
    Handles item listing with provider contact info and detailed error reporting.
    """
    if engine is None:
        return jsonify({"error": "Database engine not initialized."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    provider_name = request.form.get('provider_name', 'Anonymous')
    provider_phone = request.form.get('provider_phone', 'No Phone')
    item_title = request.form.get('item_title', 'No Title') 
    image_file = request.files['image']
    image_bytes = image_file.read()

    try:
        image_url = upload_to_gcs(image_bytes, image_file.filename)
    except Exception as e:
        return jsonify({"error": "GCS Upload Failed", "details": str(e)}), 500

    prompt = """
    You are a witty community manager for NeighborLoop. 
    Analyze this surplus item and return JSON:
    {
        "bio": "First-person witty dating-style profile bio, for the product, not longer than 2 lines",
        "category": "One-word category",
        "tags": ["tag1", "tag2"]
    }
    """
    
    try:
        response = genai_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        profile = json.loads(response.text)
        
        generated_owner_id = str(uuid.uuid4())
        
        with engine.connect() as conn:
            query = text("""
                INSERT INTO items (owner_id, provider_name, provider_phone, title, bio, category, image_url, status, item_vector)
                VALUES (:owner, :name, :phone, :title, :bio, :cat, :url, 'available', embedding('text-embedding-005', :title || ' ' || :bio)::vector) 
                RETURNING item_id
            """)
            result = conn.execute(query, {
                "owner": generated_owner_id, 
                "name": provider_name,
                "phone": provider_phone,
                #"title": profile.get('title', 'Mystery Item'),
                "title": item_title,
                "bio": profile.get('bio', 'No bio provided.'),
                "cat": profile.get('category', 'Misc'),
                "url": image_url
            })
            item_id = result.fetchone()[0]
            conn.commit()

        return jsonify({
            "status": "success",
            "item_id": str(item_id),
            "image_url": image_url,
            "profile": profile
        })

    except Exception as e:
        print(f"Error during item listing: {traceback.format_exc()}")
        return jsonify({
            "error": "Operation Failed", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/search', methods=['GET'])
def search():
    """Performs semantic vector search using pgvector."""
    if engine is None:
        return jsonify({"error": "Database engine not initialized."}), 500

    query_text = request.args.get('query') # Get the 'query' parameter from the URL
    #data = request.json
    #query_text = data.get('query')
    if not query_text:
        return jsonify([])

    # Generate vector for the search query
    '''
    query_vector = generate_embedding(query_text)
    if not query_vector:
        return jsonify({"error": "Failed to generate search embedding"}), 500
    '''
    try:
        with engine.connect() as conn:
            print(f"Searching for: {query_text}") # Log the query

            # Using Cosine Distance (<=>) for similarity
            # 1 - distance = similarity score
            search_sql = text("""
                SELECT item_id, title, bio, category, image_url, 1 - (item_vector <=> embedding('text-embedding-005', :query)::vector) as score
                FROM items 
                WHERE status = 'available' AND item_vector IS NOT NULL and 
                ai.if(
                    prompt => 'Does this text: "' || bio ||'" match the user request: "' ||  :query || '", at least 60%? " ',
                    model_id => 'gemini-3-flash-preview')  
                ORDER BY item_vector <=> embedding('text-embedding-005', :query)::vector
                LIMIT 5
            """)
            result = conn.execute(search_sql, {"query": query_text})
            
            hits = []
            for row in result:
                hits.append({
                    "id": str(row[0]),
                    "title": row[1],
                    "bio": row[2],
                    "category": row[3],
                    "image_url": row[4],
                    "score": round(float(row[5]), 3)
                })
            return jsonify(hits)
    except Exception as e:
        print(f"Error during search: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/swipe', methods=['POST'])
def handle_swipe():
    """
    Records swipe in the 'swipes' table. 
    If right swipe, updates item status and returns provider info.
    """
    if engine is None:
        return jsonify({"error": "Database engine not initialized."}), 500

    data = request.json
    direction = data.get('direction')
    item_id = data.get('item_id')
    # Generate a dummy swiper_id since we don't have login yet
    swiper_id = str(uuid.uuid4()) 

    if not item_id or direction not in ['left', 'right']:
        return jsonify({"error": "Invalid swipe data"}), 400

    try:
        with engine.connect() as conn:
            is_match = (direction == 'right')
            
            # 1. Record the swipe
            swipe_query = text("""
                INSERT INTO swipes (swiper_id, item_id, direction, is_match)
                VALUES (:swiper, :item, :dir, :match)
            """)
            conn.execute(swipe_query, {
                "swiper": swiper_id,
                "item": item_id,
                "dir": direction,
                "match": is_match
            })

            # 2. If it's a match, get provider info and mark item as 'matched'
            if is_match:
                # Fetch provider info
                info_query = text("SELECT provider_name, provider_phone FROM items WHERE item_id = :id")
                res = conn.execute(info_query, {"id": item_id}).fetchone()
                
                # Update status to remove from deck
                update_query = text("UPDATE items SET status = 'matched' WHERE item_id = :id")
                conn.execute(update_query, {"id": item_id})
                
                conn.commit()
                
                if res:
                    return jsonify({
                        "is_match": True,
                        "provider_name": res[0],
                        "provider_phone": res[1],
                        "swiper_id": swiper_id
                    })
            
            conn.commit()
            return jsonify({"is_match": False,
                "swiper_id": swiper_id
                })

    except Exception as e:
        print(f"Swipe error: {traceback.format_exc()}")
        return jsonify({
            "error": "Database error during swipe", 
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/matches', methods=['GET'])
def get_matches():
    """
    Returns a list of matches for a given swiper_id.  This is currently NOT USED.
    """
    if engine is None:
        return jsonify({"error": "Database engine not initialized."}), 500

    swiper_id = request.args.get('swiper_id')

    if not swiper_id:
        return jsonify({"error": "swiper_id is required"}), 400

    try:
        with engine.connect() as conn:
            query = text("""
                SELECT s.item_id, i.title, i.image_url, i.provider_name, i.provider_phone
                FROM swipes s
                JOIN items i ON s.item_id = i.item_id
                WHERE s.swiper_id = :swiper AND s.is_match = true AND i.status = 'matched'
            """)
            result = conn.execute(query, {"swiper": swiper_id})

            matches = []
            for row in result:
                matches.append({
                    "item_id": row[0],
                    "item_title": row[1],
                    "item_image_url": row[2],
                    "provider_name": row[3],
                    "provider_phone": row[4]
                })

            return jsonify(matches)

    except Exception as e:
        print(f"Error fetching matches: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    # Using threaded=True to handle multiple concurrent requests better
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), threaded=True)
