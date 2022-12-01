# CSE 573 Semantic Web Mining Group 21
The github repo consists of three parts
1. Model training
2. Frontend
3. Backend


# Model training
1. To train all the models download the IMDB dataset.csv file from the official source. Place the file in /input/ folder. We have not uploaded the file because of the 100 mb push resctriction on github.
2. Install all dependencies from requirements.txt file.
3. Uncomment the line 121 in the file main.py which call the function model_training
4. Run main.py using <b>python main.py</b> command. All the linear models will be trained and stored in the /models folder
5. To train the HAHNN model, 
    1. go to the /models/hahnn and run <b>pip install -r requirements.txt</b> command. 
    2. Then run <b>python -m spacy download en_core_web_md</b> command.
    3. Then run /final_code/train2.py using <b>python train2.py</b> command.
# Frontend
1. Navigate to the /client folder
2. Run npm install
3. Run <b>npm run start</b> to start the frontend on localhost:3000

# Backend
1. Start the flask server using the command <b>flask --app flask_server run</b> from the root directory.
