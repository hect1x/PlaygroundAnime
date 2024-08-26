from flask import Flask , render_template, jsonify, request
import pandas as pd
import ast, difflib
import requests, os, replicate
from dotenv import load_dotenv
import base64
#google colab link to retrieve data processing
# https://colab.research.google.com/drive/1CnBUjkGxKm2F-vaQBRA6FvpX1mhcD-5n?usp=sharing
load_dotenv()
app = Flask(__name__)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")  # Get from .env
huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

final_df = pd.read_csv('final_df.csv')

final_df['title_synonyms'] = final_df['title_synonyms'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

similarity_df = pd.read_csv('similarity_scores.csv')

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/generate")
def generate():
    return render_template('generate.html')

@app.route("/llmPage")
def llmPage():
    return render_template('llmPage.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.form.get('title').lower()


    all_titles = final_df['title'].tolist()
    closest_matches = difflib.get_close_matches(title, all_titles)

    if closest_matches:
        closest_title = closest_matches[0]
        index_match = final_df[final_df['title'] == closest_title].index[0]
    else:
        # if no close matches, check title_english and title_japanese
        match_index = final_df[
            (final_df['title_english'] == title) |
            (final_df['title_japanese'] == title)
        ].index

        if not match_index.empty: #if straight forward, 0 might be considered false
            
            index_match = match_index[0]
        else:
            index_match = None
            for index, row in final_df.iterrows():
                synonyms = row['title_synonyms']  # no more ast.literal_eval 
                if title in [syn.lower() for syn in synonyms]:
                    index_match = index
                break 
    

    if index_match is not None:
        similarity_scores = similarity_df.iloc[index_match]
        score_pairs = []
        for idx, score in similarity_scores.items():
            if idx != index_match and score > 0:  # self and obvious none
                score_pairs.append((idx, score))

        sorted_score_pairs = sorted(score_pairs, key=lambda x: x[1], reverse=True)

        top_5 = sorted_score_pairs[:5]
        similar_index = [index for index, score in top_5]

    else:
        similar_index = []
    similar_index = [int(index) for index in similar_index]
    recommendations = []
    if similar_index:
        for index in similar_index:
            recommendations.append({
                "normal_title": final_df.loc[index, 'normal_title'],
                "image_url": final_df.loc[index, 'image_url']
            })
    else:
        recommendations = None
    return jsonify(recommendations)  # ajax/jquery expects JSON files


@app.route('/generate-image', methods= ['POST'])
def generateImage():
    prompt = request.form.get('prompt')

   
    api_url = "https://api-inference.huggingface.co/models/Ojimi/anime-kawai-diffusion"

    headers = {
        "Authorization": f"Bearer {huggingface_api_token}"
    }

    data = {
        "inputs": prompt,
        "options": {
            "use_gpu": True  #huggingface's
        }
    }

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        image_data = response.content
        img_str = base64.b64encode(image_data).decode()
        return jsonify({"image": img_str})
    else:
        print('Error from Hugging Face API:', response.status_code, response.text)
        return jsonify({"error": "Failed to generate image"}), 500

@app.route('/generate-text', methods= ['POST'])
def generateText():
    pre_prompt = "You are a helpful assistant. You specialize in stuff like anime. If 'user' asks for something outside of anime, come up with a topic to go back to anime"
    prompt = request.form.get('prompt')
    # print(prompt)
    output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
                        input={"prompt": f"{pre_prompt} {prompt} Assistant: ", 
                        "temperature":0.1, "top_p":0.9, "max_length":128, "repetition_penalty":1})  
    # print(output)
    full_response = ""

    for item in output:
        full_response += item

    return jsonify({"text" : full_response})

if __name__ == '__main__':
    app.run(debug=True)