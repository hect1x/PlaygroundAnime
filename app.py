from flask import Flask , render_template, jsonify, request
import pandas as pd
import ast, difflib

#google colab link to retrieve data processing
# https://colab.research.google.com/drive/1CnBUjkGxKm2F-vaQBRA6FvpX1mhcD-5n?usp=sharing
app = Flask(__name__)


final_df = pd.read_csv('final_df.csv')

final_df['title_synonyms'] = final_df['title_synonyms'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

similarity_df = pd.read_csv('similarity_scores.csv')

@app.route("/")

def index():
    return render_template('index.html')

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

if __name__ == '__main__':
    app.run(debug=True)